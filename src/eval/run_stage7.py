"""Stage 7: Run Experiments E6 (Transfer), E7 (Calibration), E8 (Ablations), E9 (Retrieval Quality), E10 (Failure Breakdown), and optional E11."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from src.agent.loop import LLMGenerator, run_agent_batch
from src.agent.readiness_head import ReadinessHead
from src.baselines.confidence_rule import sweep_confidence_thresholds
from src.baselines.fixed_limit import sweep_fixed_limits
from src.baselines.own_decision import evaluate_own_decision
from src.eval.calibration import compute_calibration_metrics
from src.eval.evaluate import evaluate_online_runs_against_hindsight
from src.eval.failure_taxonomy import analyze_failures
from src.eval.metrics import summarize_policy_records
from src.eval.stats import paired_tests
from src.tools.retriever import HybridRetriever
from src.train.grpo import train_grpo_model
from src.train.sft import build_sft_examples_from_runs, train_sft_model
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger, write_jsonl


def run_stage7(
    config_path: str = "configs/default.yaml",
    smoke: bool = False,
    test_limit_per_domain: int = 100,
    run_e11_moe: bool = True,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["primary_seed"])
    set_seed(seed)

    data_dir = Path(cfg["project"]["data_dir"])
    splits_dir = data_dir / ("splits_smoke" if smoke else "splits")
    index_dir = data_dir / ("index_smoke" if smoke else "index")
    runs_dir = Path(cfg["project"]["runs_dir"]) / ("smoke" if smoke else "full")
    ckpt_root = Path(cfg["project"]["checkpoints_dir"]) / ("smoke" if smoke else "full")
    results_dir = Path(cfg["project"]["output_dir"])
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("stage7", results_dir / "logs" / "stage7_experiments_e6_to_e11.log")
    prefix = "smoke_" if smoke else ""
    if smoke:
        test_limit_per_domain = 10

    domains = cfg["datasets"]["domains"]
    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_name=cfg["retriever"]["embedding_model"],
        reranker_model_name=cfg["retriever"]["reranker_model"],
        device="cuda",
    )
    retriever.build_or_load(force_rebuild=False)

    # Load domain SFT examples and reference hindsight maps
    domain_sft_train: Dict[str, List[Dict[str, Any]]] = {}
    domain_sft_val: Dict[str, List[Dict[str, Any]]] = {}
    ref_hindsight_by_domain: Dict[str, Dict[str, int]] = {}
    fl_curve_by_domain: Dict[str, List[Tuple[float, float]]] = {}
    best_fl_recs_by_domain: Dict[str, List[Dict[str, Any]]] = {}

    for domain in domains:
        tr_tasks = {t["task_id"]: t for t in read_jsonl(splits_dir / f"{domain}_train.jsonl")}
        val_tasks = {t["task_id"]: t for t in read_jsonl(splits_dir / f"{domain}_val.jsonl")}
        tr_lab = read_jsonl(runs_dir / f"{domain}_train_labelled.jsonl")
        val_lab = read_jsonl(runs_dir / f"{domain}_val_labelled.jsonl")
        test_lab = read_jsonl(runs_dir / f"{domain}_test_labelled.jsonl")

        domain_sft_train[domain] = build_sft_examples_from_runs(tr_lab, tr_tasks, balance_classes=True, seed=seed)
        domain_sft_val[domain] = build_sft_examples_from_runs(val_lab, val_tasks, balance_classes=False, seed=seed)

        ref_h: Dict[str, int] = {}
        for r in test_lab:
            hs = r.get("hindsight", {}).get("first_ready_step")
            if hs is not None:
                ref_h[r["task_id"]] = min(ref_h.get(r["task_id"], 99), int(hs))
        ref_hindsight_by_domain[domain] = ref_h

        fl_sweep = sweep_fixed_limits(test_lab, limits=cfg["eval"]["fixed_limits"])
        fl_curve_by_domain[domain] = [
            (
                summarize_policy_records(item["records"], seed=seed)["mean_steps"],
                summarize_policy_records(item["records"], seed=seed)["accuracy"],
            )
            for item in fl_sweep
        ]
        best_fl_recs_by_domain[domain] = max(
            (item["records"] for item in fl_sweep),
            key=lambda recs: summarize_policy_records(recs, seed=seed)["accuracy"],
        )

    # =========================================================================
    # E6: Cross-Domain Transfer (Train on 2 domains, test on held-out 3rd domain)
    # =========================================================================
    logger.info("Running E6: Cross-domain stopping transfer...")
    e6_rows: List[Dict[str, Any]] = []
    for held_out in domains:
        source_domains = [d for d in domains if d != held_out]
        transfer_train_ex = []
        for sd in source_domains:
            transfer_train_ex.extend(domain_sft_train[sd])
        transfer_val_ex = domain_sft_val[held_out] or transfer_train_ex[-20:]

        transfer_ckpt = ckpt_root / f"transfer_holdout_{held_out}"
        train_sft_model(
            train_examples=transfer_train_ex,
            val_examples=transfer_val_ex,
            output_dir=transfer_ckpt,
            base_model_name=cfg["model"]["base_model"],
            epochs=1 if smoke else 2,
            lr=float(cfg["sft"]["learning_rate"]),
            batch_size=int(cfg["sft"]["batch_size"]),
            grad_accum_steps=int(cfg["sft"]["grad_accum_steps"]),
            seed=seed,
        )

        test_tasks = read_jsonl(splits_dir / f"{held_out}_test.jsonl")[:test_limit_per_domain]
        out_transfer_runs = runs_dir / f"{held_out}_test_transfer_runs.jsonl"
        if out_transfer_runs.exists():
            tr_runs = read_jsonl(out_transfer_runs)
        else:
            gen = LLMGenerator(
                model_name=cfg["model"]["base_model"],
                adapter_path=str(transfer_ckpt),
                device="cuda",
                dtype=cfg["model"]["dtype"],
            )
            tr_runs = run_agent_batch(
                tasks=test_tasks,
                generator=gen,
                retriever=retriever,
                max_steps=7,
                temperature=0.2,
                few_shot=True,
                include_readiness=True,
                run_id_prefix=f"transfer_to_{held_out}",
                sample_idx=0,
                batch_size=32,
            )
            write_jsonl(out_transfer_runs, tr_runs)
            del gen
            torch.cuda.empty_cache()

        summ_tr, recs_tr = evaluate_online_runs_against_hindsight(
            tr_runs,
            reference_hindsight_map=ref_hindsight_by_domain[held_out],
            fixed_limit_curve=fl_curve_by_domain[held_out],
            seed=seed,
        )
        # Compare against in-domain Method A and Untrained Own Decision on held_out
        indomain_runs = read_jsonl(runs_dir / f"{held_out}_test_method_a_runs.jsonl")
        summ_in, recs_in = evaluate_online_runs_against_hindsight(
            indomain_runs,
            reference_hindsight_map=ref_hindsight_by_domain[held_out],
            fixed_limit_curve=fl_curve_by_domain[held_out],
            seed=seed,
        )
        base_lab = read_jsonl(runs_dir / f"{held_out}_test_labelled.jsonl")
        own_recs = evaluate_own_decision(base_lab)
        summ_own = summarize_policy_records(own_recs, fixed_limit_curve=fl_curve_by_domain[held_out], seed=seed)

        # Paired test: Transfer vs Own Decision (first sample per task)
        own_by_task = {}
        for r in own_recs:
            if r["task_id"] not in own_by_task:
                own_by_task[r["task_id"]] = r["correct"]
        tr_accs = [r["correct"] for r in recs_tr if r["task_id"] in own_by_task]
        own_accs = [own_by_task[r["task_id"]] for r in recs_tr if r["task_id"] in own_by_task]
        pt_vs_own = paired_tests(tr_accs, own_accs, seed=seed)

        e6_rows.append(
            {
                "train_domains": "+".join(source_domains),
                "test_domain": held_out,
                "setting": "Out-of-Domain Transfer (SFT)",
                **summ_tr,
                **pt_vs_own,
                "indomain_sft_accuracy": summ_in["accuracy"],
                "indomain_sft_steps": summ_in["mean_steps"],
                "untrained_own_accuracy": summ_own["accuracy"],
                "untrained_own_steps": summ_own["mean_steps"],
            }
        )
    pd.DataFrame(e6_rows).to_csv(results_dir / f"{prefix}e6_transfer.csv", index=False)

    # =========================================================================
    # E7: Readiness Score Calibration (Reliability Diagram, ECE, Threshold Sweep, Linear ReadinessHead)
    # =========================================================================
    logger.info("Running E7: Readiness calibration analysis...")
    e7_summary_rows: List[Dict[str, Any]] = []
    e7_bin_rows: List[Dict[str, Any]] = []
    e7_thresh_rows: List[Dict[str, Any]] = []

    # 1. Untrained Base Agent cut-off calibration across all test runs
    base_preds: List[float] = []
    base_trues: List[float] = []
    all_base_test_labelled: List[Dict[str, Any]] = []
    for d in domains:
        d_runs = read_jsonl(runs_dir / f"{d}_test_labelled.jsonl")
        all_base_test_labelled.extend(d_runs)
        for r in d_runs:
            steps = r.get("steps", [])
            cutoffs = {int(c["step_index"]): float(c["ready_fraction"]) for c in r.get("cutoff_labels", [])}
            for s in steps:
                s_idx = int(s["step_index"])
                pred_r = float(s.get("readiness", 0.5))
                if s_idx in cutoffs:
                    true_r = float(cutoffs[s_idx] >= 0.6)
                else:
                    true_r = float(r.get("correct", 0))
                base_preds.append(pred_r)
                base_trues.append(true_r)

    cal_base = compute_calibration_metrics(base_preds, base_trues, n_bins=10)
    e7_summary_rows.append({"model": "Untrained Base Agent (Text Readiness)", "ece": cal_base["ece"], "brier": cal_base["brier"], "n_samples": cal_base["n_samples"]})
    for b in cal_base["bins"]:
        e7_bin_rows.append({"model": "Untrained Base Agent", **b})

    # 2. Method A (SFT) & Method C (SFT+GRPO) calibration on test runs
    for model_label, run_suffix in [
        ("Method A (SFT Text Readiness)", "method_a_runs.jsonl"),
        ("Method C (SFT+GRPO Text Readiness)", "method_c_lam_0p1_runs.jsonl"),
    ]:
        m_preds: List[float] = []
        m_trues: List[float] = []
        for d in domains:
            p_file = runs_dir / f"{d}_test_{run_suffix}"
            if not p_file.exists():
                continue
            m_runs = read_jsonl(p_file)
            ref_h = ref_hindsight_by_domain[d]
            for r in m_runs:
                tid = r["task_id"]
                h_step = ref_h.get(tid)
                for s in r.get("steps", []):
                    s_idx = int(s["step_index"])
                    pred_r = float(s.get("readiness", 0.5))
                    if s.get("action") == "ANSWER":
                        true_r = float(r.get("correct", 0))
                    elif h_step is not None:
                        true_r = float(s_idx >= h_step)
                    else:
                        true_r = float(r.get("correct", 0) and s_idx >= r.get("num_steps", 1))
                    m_preds.append(pred_r)
                    m_trues.append(true_r)
        if m_preds:
            cal_m = compute_calibration_metrics(m_preds, m_trues, n_bins=10)
            e7_summary_rows.append({"model": model_label, "ece": cal_m["ece"], "brier": cal_m["brier"], "n_samples": cal_m["n_samples"]})
            short_name = "Method A (SFT)" if "Method A" in model_label else "Method C (A+B)"
            for b in cal_m["bins"]:
                e7_bin_rows.append({"model": short_name, **b})

    # 3. Train & evaluate auxiliary Linear ReadinessHead (on feature representations of step states)
    # Using TF-IDF + SVD feature projection into ReadinessHead (torch.nn.Module) trained with BCE loss on train cut-off labels
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_texts_rh: List[str] = []
    train_targets_rh: List[float] = []
    for d in domains:
        for r in read_jsonl(runs_dir / f"{d}_train_labelled.jsonl"):
            cutoffs = {int(c["step_index"]): float(c["ready_fraction"]) for c in r.get("cutoff_labels", [])}
            for s in r.get("steps", []):
                s_idx = int(s["step_index"])
                st_txt = f"{s.get('think', '')} {s.get('action', '')} {s.get('tool_result', '')} ready={s.get('readiness', 0.5)}"
                y_val = float(cutoffs[s_idx] >= 0.6) if s_idx in cutoffs else float(r.get("correct", 0))
                train_texts_rh.append(st_txt)
                train_targets_rh.append(y_val)

    test_texts_rh: List[str] = []
    test_targets_rh: List[float] = []
    for r in all_base_test_labelled:
        cutoffs = {int(c["step_index"]): float(c["ready_fraction"]) for c in r.get("cutoff_labels", [])}
        for s in r.get("steps", []):
            s_idx = int(s["step_index"])
            st_txt = f"{s.get('think', '')} {s.get('action', '')} {s.get('tool_result', '')} ready={s.get('readiness', 0.5)}"
            y_val = float(cutoffs[s_idx] >= 0.6) if s_idx in cutoffs else float(r.get("correct", 0))
            test_texts_rh.append(st_txt)
            test_targets_rh.append(y_val)

    if train_texts_rh and test_texts_rh:
        vec = TfidfVectorizer(max_features=256)
        X_tr = torch.tensor(vec.fit_transform(train_texts_rh).toarray(), dtype=torch.float32)
        y_tr = torch.tensor(train_targets_rh, dtype=torch.float32)
        X_te = torch.tensor(vec.transform(test_texts_rh).toarray(), dtype=torch.float32)

        head = ReadinessHead(hidden_size=X_tr.shape[1])
        opt_h = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
        head.train()
        for _ in range(150):
            opt_h.zero_grad()
            preds_tr = head(X_tr)
            loss_h = ReadinessHead.compute_loss(preds_tr, y_tr)
            loss_h.backward()
            opt_h.step()
        head.eval()
        with torch.no_grad():
            rh_preds = head(X_te).tolist()
        cal_rh = compute_calibration_metrics(rh_preds, test_targets_rh, n_bins=10)
        e7_summary_rows.append({"model": "Trained ReadinessHead (Linear Probe)", "ece": cal_rh["ece"], "brier": cal_rh["brier"], "n_samples": cal_rh["n_samples"]})
        for b in cal_rh["bins"]:
            e7_bin_rows.append({"model": "ReadinessHead Probe", **b})

    # Threshold sweep for early-stop / over-run rates when stopping by threshold on readiness score
    thresh_sweep = sweep_confidence_thresholds(
        all_base_test_labelled,
        thresholds=cfg["eval"]["confidence_thresholds"],
        score_key="readiness",
    )
    for item in thresh_sweep:
        s_t = summarize_policy_records(item["records"], seed=seed)
        e7_thresh_rows.append({"threshold": item["threshold"], **s_t})

    pd.DataFrame(e7_summary_rows).to_csv(results_dir / f"{prefix}e7_calibration_summary.csv", index=False)
    pd.DataFrame(e7_bin_rows).to_csv(results_dir / f"{prefix}e7_calibration_bins.csv", index=False)
    pd.DataFrame(e7_thresh_rows).to_csv(results_dir / f"{prefix}e7_threshold_sweep.csv", index=False)

    # =========================================================================
    # E8: Ablations (Method A ablations + Method B reward ablations)
    # =========================================================================
    logger.info("Running E8: Method A and Method B ablations...")
    e8_rows: List[Dict[str, Any]] = []
    ablation_eval_domain = "search_qa"
    ab_test_tasks = read_jsonl(splits_dir / f"{ablation_eval_domain}_test.jsonl")[:test_limit_per_domain]
    ab_ref_h = ref_hindsight_by_domain[ablation_eval_domain]
    ab_fl_curve = fl_curve_by_domain[ablation_eval_domain]

    # Reference Method A Full on search_qa
    ma_full_runs = read_jsonl(runs_dir / f"{ablation_eval_domain}_test_method_a_runs.jsonl")
    summ_ma_full, recs_ma_full = evaluate_online_runs_against_hindsight(
        ma_full_runs, reference_hindsight_map=ab_ref_h, fixed_limit_curve=ab_fl_curve, seed=seed
    )
    e8_rows.append({"ablation": "Method A (Full: Balanced + Readiness + All Cut-offs)", "group": "Method A", **summ_ma_full})

    # Three Method A ablations:
    # (1) no readiness output (`include_readiness=False`)
    # (2) no class balancing (`balance_classes=False`)
    # (3) only 3 cut-off points per run (`max_cutoff_points=3`)
    for ab_slug, ab_label, bal_flag, rdy_flag, max_cp in [
        ("no_readiness", "Method A - No Readiness Output", True, False, None),
        ("no_balance", "Method A - No Class Balancing", False, True, None),
        ("3_cutoffs", "Method A - Only 3 Cut-off Points/Run", True, True, 3),
    ]:
        ab_ex: List[Dict[str, Any]] = []
        for d in domains:
            t_map = {t["task_id"]: t for t in read_jsonl(splits_dir / f"{d}_train.jsonl")}
            t_lab = read_jsonl(runs_dir / f"{d}_train_labelled.jsonl")
            ab_ex.extend(
                build_sft_examples_from_runs(
                    t_lab,
                    t_map,
                    balance_classes=bal_flag,
                    include_readiness=rdy_flag,
                    max_cutoff_points=max_cp,
                    seed=seed,
                )
            )
        ab_ckpt = ckpt_root / f"ablation_sft_{ab_slug}"
        train_sft_model(
            train_examples=ab_ex,
            val_examples=domain_sft_val[ablation_eval_domain][:20],
            output_dir=ab_ckpt,
            base_model_name=cfg["model"]["base_model"],
            epochs=1 if smoke else 2,
            lr=float(cfg["sft"]["learning_rate"]),
            batch_size=int(cfg["sft"]["batch_size"]),
            grad_accum_steps=int(cfg["sft"]["grad_accum_steps"]),
            seed=seed,
        )
        ab_runs_file = runs_dir / f"{ablation_eval_domain}_test_ablation_{ab_slug}.jsonl"
        if ab_runs_file.exists():
            ab_runs = read_jsonl(ab_runs_file)
        else:
            gen = LLMGenerator(
                model_name=cfg["model"]["base_model"],
                adapter_path=str(ab_ckpt),
                device="cuda",
                dtype=cfg["model"]["dtype"],
            )
            ab_runs = run_agent_batch(
                tasks=ab_test_tasks,
                generator=gen,
                retriever=retriever,
                max_steps=7,
                temperature=0.2,
                few_shot=True,
                include_readiness=rdy_flag,
                run_id_prefix=f"ab_{ab_slug}",
                sample_idx=0,
                batch_size=32,
            )
            write_jsonl(ab_runs_file, ab_runs)
            del gen
            torch.cuda.empty_cache()

        summ_ab, recs_ab = evaluate_online_runs_against_hindsight(
            ab_runs, reference_hindsight_map=ab_ref_h, fixed_limit_curve=ab_fl_curve, seed=seed
        )
        summ_ab.update(
            paired_tests(
                [r["correct"] for r in recs_ab],
                [r["correct"] for r in recs_ma_full],
                seed=seed,
            )
        )
        e8_rows.append({"ablation": ab_label, "group": "Method A", **summ_ab})

    # Method B reward ablations:
    # (1) Full reward (lambda=0.1) -> from Stage 6 `method_b_lam_0p1`
    # (2) Reward without effort term (`effort_mode="none"` / `lam=0.0`) -> from Stage 6 `method_b_lam_0p0`
    # (3) Reward using step count only (`effort_mode="steps_only"`, `lam=0.1`)
    mb_full_runs = read_jsonl(runs_dir / f"{ablation_eval_domain}_test_method_b_lam_0p1_runs.jsonl")
    summ_mb_full, recs_mb_full = evaluate_online_runs_against_hindsight(
        mb_full_runs, reference_hindsight_map=ab_ref_h, fixed_limit_curve=ab_fl_curve, seed=seed
    )
    e8_rows.append({"ablation": "Method B (Full Reward: Step + Token Effort)", "group": "Method B", **summ_mb_full})

    mb_noeff_runs = read_jsonl(runs_dir / f"{ablation_eval_domain}_test_method_b_lam_0p0_runs.jsonl")
    summ_mb_noeff, recs_mb_noeff = evaluate_online_runs_against_hindsight(
        mb_noeff_runs, reference_hindsight_map=ab_ref_h, fixed_limit_curve=ab_fl_curve, seed=seed
    )
    summ_mb_noeff.update(
        paired_tests([r["correct"] for r in recs_mb_noeff], [r["correct"] for r in recs_mb_full], seed=seed)
    )
    e8_rows.append({"ablation": "Method B - Reward w/o Effort Term (lambda=0)", "group": "Method B", **summ_mb_noeff})

    steps_only_ckpt = ckpt_root / "ablation_grpo_steps_only"
    interleaved_tr = read_jsonl(splits_dir / f"{ablation_eval_domain}_train.jsonl")[:100]
    train_grpo_model(
        train_tasks=interleaved_tr,
        val_tasks=[],
        output_dir=steps_only_ckpt,
        retriever=retriever,
        base_model_name=cfg["model"]["base_model"],
        init_adapter_path=None,
        lam=0.1,
        group_size=4 if smoke else 6,
        kl_coef=float(cfg["grpo"]["kl_coef"]),
        lr=float(cfg["grpo"]["learning_rate"]),
        max_steps=10 if smoke else 40,
        save_every=20,
        agent_max_steps=5,
        effort_mode="steps_only",
        seed=seed,
    )
    so_runs_file = runs_dir / f"{ablation_eval_domain}_test_ablation_steps_only.jsonl"
    if so_runs_file.exists():
        so_runs = read_jsonl(so_runs_file)
    else:
        gen = LLMGenerator(
            model_name=cfg["model"]["base_model"],
            adapter_path=str(steps_only_ckpt),
            device="cuda",
            dtype=cfg["model"]["dtype"],
        )
        so_runs = run_agent_batch(
            tasks=ab_test_tasks,
            generator=gen,
            retriever=retriever,
            max_steps=7,
            temperature=0.2,
            few_shot=True,
            include_readiness=True,
            run_id_prefix="ab_steps_only",
            sample_idx=0,
            batch_size=32,
        )
        write_jsonl(so_runs_file, so_runs)
        del gen
        torch.cuda.empty_cache()

    summ_so, recs_so = evaluate_online_runs_against_hindsight(
        so_runs, reference_hindsight_map=ab_ref_h, fixed_limit_curve=ab_fl_curve, seed=seed
    )
    summ_so.update(paired_tests([r["correct"] for r in recs_so], [r["correct"] for r in recs_mb_full], seed=seed))
    e8_rows.append({"ablation": "Method B - Reward w/ Step Count Only", "group": "Method B", **summ_so})

    pd.DataFrame(e8_rows).to_csv(results_dir / f"{prefix}e8_ablations.csv", index=False)

    # =========================================================================
    # E9: Retrieval Quality Adaptation on Search QA (no re-ranker; k=1 vs 3 vs 5)
    # =========================================================================
    logger.info("Running E9: Retrieval quality sensitivity on Search QA...")
    e9_rows: List[Dict[str, Any]] = []
    sqa_test_tasks = read_jsonl(splits_dir / "search_qa_test.jsonl")[:test_limit_per_domain]
    best_policy_ckpt = ckpt_root / "method_c_lam_0p1"
    if not best_policy_ckpt.exists():
        best_policy_ckpt = ckpt_root / "method_a_sft"

    gen_policy = LLMGenerator(
        model_name=cfg["model"]["base_model"],
        adapter_path=str(best_policy_ckpt),
        device="cuda",
        dtype=cfg["model"]["dtype"],
    )

    for r_slug, r_label, top_k_val, rerank_flag in [
        ("k3_rerank", "k=3, w/ Cross-Encoder Re-ranker (Default)", 3, True),
        ("k3_norerank", "k=3, No Re-ranker (BM25+FAISS RRF only)", 3, False),
        ("k1_rerank", "k=1, w/ Cross-Encoder Re-ranker", 1, True),
        ("k5_rerank", "k=5, w/ Cross-Encoder Re-ranker", 5, True),
    ]:
        e9_file = runs_dir / f"search_qa_test_e9_{r_slug}.jsonl"
        if e9_file.exists():
            r_runs = read_jsonl(e9_file)
        else:
            r_runs = run_agent_batch(
                tasks=sqa_test_tasks,
                generator=gen_policy,
                retriever=retriever,
                max_steps=7,
                temperature=0.2,
                few_shot=True,
                include_readiness=True,
                top_k=top_k_val,
                use_reranker=rerank_flag,
                run_id_prefix=f"e9_{r_slug}",
                sample_idx=0,
                batch_size=32,
            )
            write_jsonl(e9_file, r_runs)

        summ_r, _ = evaluate_online_runs_against_hindsight(
            r_runs,
            reference_hindsight_map=ref_hindsight_by_domain["search_qa"],
            fixed_limit_curve=fl_curve_by_domain["search_qa"],
            seed=seed,
        )
        e9_rows.append(
            {
                "retrieval_setting": r_label,
                "top_k": top_k_val,
                "use_reranker": rerank_flag,
                **summ_r,
            }
        )
    del gen_policy
    torch.cuda.empty_cache()
    pd.DataFrame(e9_rows).to_csv(results_dir / f"{prefix}e9_retrieval_quality.csv", index=False)

    # =========================================================================
    # E10: Failure Taxonomy & 30-Example Manual/Rule Spot Check
    # =========================================================================
    logger.info("Running E10: Failure breakdown across methods...")
    e10_summary_rows: List[Dict[str, Any]] = []
    spot_check_candidates: List[Dict[str, Any]] = []

    method_run_collections: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("Own Decision (Untrained)", all_base_test_labelled),
    ]
    for m_label, f_suffix in [
        ("Method A (SFT)", "method_a_runs.jsonl"),
        ("Method B (GRPO lam=0.1)", "method_b_lam_0p1_runs.jsonl"),
        ("Method B (GRPO lam=0.6)", "method_b_lam_0p6_runs.jsonl"),
        ("Method C (SFT+GRPO lam=0.1)", "method_c_lam_0p1_runs.jsonl"),
    ]:
        coll: List[Dict[str, Any]] = []
        for d in domains:
            pf = runs_dir / f"{d}_test_{f_suffix}"
            if pf.exists():
                coll.extend(read_jsonl(pf))
        if coll:
            method_run_collections.append((m_label, coll))

    for m_label, coll_runs in method_run_collections:
        f_info = analyze_failures(coll_runs, method_name=m_label, max_steps=7)
        e10_summary_rows.append(f_info["summary"])
        if m_label == "Own Decision (Untrained)":
            spot_check_candidates.extend(f_info["examples"])

    pd.DataFrame(e10_summary_rows).to_csv(results_dir / f"{prefix}e10_failure_taxonomy.csv", index=False)
    pd.DataFrame(spot_check_candidates[:30]).to_csv(results_dir / f"{prefix}e10_spot_check_30.csv", index=False)

    # =========================================================================
    # E11 (Optional): Scale / Architecture Comparison (Qwen2.5-7B-Instruct vs 3B)
    # =========================================================================
    e11_rows: List[Dict[str, Any]] = []
    if run_e11_moe and not smoke:
        try:
            logger.info("Running E11: Comparing Qwen2.5-3B-Instruct (Method C) vs cached Qwen2.5-7B-Instruct...")
            gen_7b = LLMGenerator(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                adapter_path=None,
                device="cuda",
                dtype="bfloat16",
            )
            sqa_slice = sqa_test_tasks[:50]
            e11_file = runs_dir / "search_qa_test_e11_qwen7b.jsonl"
            if e11_file.exists():
                runs_7b = read_jsonl(e11_file)
            else:
                runs_7b = run_agent_batch(
                    tasks=sqa_slice,
                    generator=gen_7b,
                    retriever=retriever,
                    max_steps=7,
                    temperature=0.2,
                    few_shot=True,
                    include_readiness=True,
                    run_id_prefix="e11_7b",
                    sample_idx=0,
                    batch_size=16,
                )
                write_jsonl(e11_file, runs_7b)
            del gen_7b
            torch.cuda.empty_cache()

            summ_7b, _ = evaluate_online_runs_against_hindsight(
                runs_7b,
                reference_hindsight_map=ref_hindsight_by_domain["search_qa"],
                fixed_limit_curve=fl_curve_by_domain["search_qa"],
                seed=seed,
            )
            e11_rows.append({"model": "Qwen2.5-7B-Instruct (Untrained Own Decision)", "domain": "search_qa", **summ_7b})

            c_3b_runs = read_jsonl(runs_dir / "search_qa_test_method_c_lam_0p1_runs.jsonl")[:50]
            summ_3b_c, _ = evaluate_online_runs_against_hindsight(
                c_3b_runs,
                reference_hindsight_map=ref_hindsight_by_domain["search_qa"],
                fixed_limit_curve=fl_curve_by_domain["search_qa"],
                seed=seed,
            )
            e11_rows.append({"model": "Qwen2.5-3B-Instruct + StopWise Method C (Ours)", "domain": "search_qa", **summ_3b_c})
            pd.DataFrame(e11_rows).to_csv(results_dir / f"{prefix}e11_model_comparison.csv", index=False)
        except Exception as e:
            logger.warning(f"Optional E11 comparison skipped or partial: {e}")

    out = {
        "smoke": smoke,
        "e6_transfer": e6_rows,
        "e7_calibration": e7_summary_rows,
        "e8_ablations": e8_rows,
        "e9_retrieval": e9_rows,
        "e10_failures": e10_summary_rows,
        "e11_comparison": e11_rows,
    }
    with open(results_dir / f"{prefix}stage7_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Stage 7 complete.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_stage7(config_path=args.config, smoke=args.smoke)

