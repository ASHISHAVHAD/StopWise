"""Stage 6: Train Methods B and C (GRPO across lambda in {0, 0.1, 0.3, 0.6}) and evaluate E4 & E5."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.agent.loop import LLMGenerator, run_agent_batch
from src.baselines.fixed_limit import sweep_fixed_limits
from src.baselines.hindsight_best import evaluate_hindsight_best
from src.eval.evaluate import evaluate_online_runs_against_hindsight
from src.eval.metrics import compute_auc_acc_steps, summarize_policy_records
from src.eval.stats import paired_tests
from src.tools.retriever import HybridRetriever
from src.train.grpo import train_grpo_model
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger, write_jsonl


def run_stage6(
    config_path: str = "configs/default.yaml",
    smoke: bool = False,
    grpo_steps: int = 100,
    test_limit_per_domain: int = 100,
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

    logger = setup_logger("stage6", results_dir / "logs" / "stage6_grpo_b_c.log")
    prefix = "smoke_" if smoke else ""

    if smoke:
        grpo_steps = 12
        test_limit_per_domain = 10
        group_size = 4
        save_every = 6
    else:
        group_size = int(cfg["grpo"]["group_size"])
        save_every = max(25, grpo_steps // 4)

    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_name=cfg["retriever"]["embedding_model"],
        reranker_model_name=cfg["retriever"]["reranker_model"],
        device="cuda",
    )
    retriever.build_or_load(force_rebuild=False)

    domains = cfg["datasets"]["domains"]
    # Interleave training tasks from all 3 domains so GRPO trains a multi-domain stopping policy
    domain_train_lists = [read_jsonl(splits_dir / f"{d}_train.jsonl")[:200] for d in domains]
    interleaved_train: List[Dict[str, Any]] = []
    max_len = max(len(x) for x in domain_train_lists)
    for i in range(max_len):
        for d_list in domain_train_lists:
            if i < len(d_list):
                interleaved_train.append(d_list[i])

    val_tasks: List[Dict[str, Any]] = []
    for d in domains:
        val_tasks.extend(read_jsonl(splits_dir / f"{d}_val.jsonl")[:15])

    lambdas = [float(x) for x in cfg["grpo"]["lambdas"]]
    method_a_ckpt = ckpt_root / "method_a_sft"

    t0 = time.time()
    grpo_train_summaries: Dict[str, Any] = {}

    # 1. Train Method B (from base model) for each lambda in {0.0, 0.1, 0.3, 0.6}
    for lam in lambdas:
        lam_tag = str(lam).replace(".", "p")
        b_ckpt = ckpt_root / f"method_b_lam_{lam_tag}"
        logger.info(f"Training Method B (GRPO from base) with lambda={lam} ({grpo_steps} steps) -> {b_ckpt}...")
        b_metrics = train_grpo_model(
            train_tasks=interleaved_train,
            val_tasks=val_tasks,
            output_dir=b_ckpt,
            retriever=retriever,
            base_model_name=cfg["model"]["base_model"],
            init_adapter_path=None,
            lam=lam,
            group_size=group_size,
            kl_coef=float(cfg["grpo"]["kl_coef"]),
            lr=float(cfg["grpo"]["learning_rate"]),
            max_steps=grpo_steps,
            save_every=save_every,
            agent_max_steps=5,
            seed=seed,
        )
        grpo_train_summaries[f"method_b_lam_{lam}"] = b_metrics

    # 2. Train Method C (starting from Method A SFT checkpoint) for lambda in {0.1, 0.3}
    method_c_lambdas = [0.1, 0.3] if not smoke else [0.1]
    for lam in method_c_lambdas:
        lam_tag = str(lam).replace(".", "p")
        c_ckpt = ckpt_root / f"method_c_lam_{lam_tag}"
        logger.info(f"Training Method C (SFT -> GRPO) with lambda={lam} ({grpo_steps} steps) -> {c_ckpt}...")
        c_metrics = train_grpo_model(
            train_tasks=interleaved_train,
            val_tasks=val_tasks,
            output_dir=c_ckpt,
            retriever=retriever,
            base_model_name=cfg["model"]["base_model"],
            init_adapter_path=method_a_ckpt,
            lam=lam,
            group_size=group_size,
            kl_coef=float(cfg["grpo"]["kl_coef"]),
            lr=float(cfg["grpo"]["learning_rate"]),
            max_steps=grpo_steps,
            save_every=save_every,
            agent_max_steps=5,
            seed=seed,
        )
        grpo_train_summaries[f"method_c_lam_{lam}"] = c_metrics

    # 3. Evaluate Method B (E4) and Method C (E5) on held-out test sets per domain
    e4_rows: List[Dict[str, Any]] = []
    e5_rows: List[Dict[str, Any]] = []

    # Cache generators by checkpoint path so we load each adapter once across all 3 domains
    eval_models: List[Tuple[str, str, float, Path]] = []
    for lam in lambdas:
        lam_tag = str(lam).replace(".", "p")
        eval_models.append(("Method B (GRPO)", f"lam_{lam}", lam, ckpt_root / f"method_b_lam_{lam_tag}"))
    for lam in method_c_lambdas:
        lam_tag = str(lam).replace(".", "p")
        eval_models.append(("Method C (A+B)", f"lam_{lam}", lam, ckpt_root / f"method_c_lam_{lam_tag}"))

    runs_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for method_label, setting_str, lam_val, ckpt_dir in eval_models:
        m_slug = "method_b" if "Method B" in method_label else "method_c"
        lam_tag = str(lam_val).replace(".", "p")
        gen = LLMGenerator(
            model_name=cfg["model"]["base_model"],
            adapter_path=str(ckpt_dir),
            device="cuda",
            dtype=cfg["model"]["dtype"],
        )
        for domain in domains:
            test_tasks = read_jsonl(splits_dir / f"{domain}_test.jsonl")[:test_limit_per_domain]
            out_file = runs_dir / f"{domain}_test_{m_slug}_lam_{lam_tag}_runs.jsonl"
            if out_file.exists():
                m_runs = read_jsonl(out_file)
            else:
                logger.info(f"Evaluating {method_label} ({setting_str}) on {domain} ({len(test_tasks)} tasks)...")
                m_runs = run_agent_batch(
                    tasks=test_tasks,
                    generator=gen,
                    retriever=retriever,
                    max_steps=7,
                    temperature=0.2,
                    few_shot=True,
                    include_readiness=True,
                    run_id_prefix=f"{m_slug}_{lam_tag}",
                    sample_idx=0,
                    batch_size=32,
                )
                write_jsonl(out_file, m_runs)
            runs_cache[(m_slug, setting_str, domain)] = m_runs
        del gen
        import torch
        torch.cuda.empty_cache()

    # Compute E4 and E5 tables per domain
    for domain in domains:
        base_test_labelled = read_jsonl(runs_dir / f"{domain}_test_labelled.jsonl")
        ref_hindsight: Dict[str, int] = {}
        for r in base_test_labelled:
            hs = r.get("hindsight", {}).get("first_ready_step")
            if hs is not None:
                tid = r["task_id"]
                ref_hindsight[tid] = min(ref_hindsight.get(tid, 99), int(hs))

        fl_sweep = sweep_fixed_limits(base_test_labelled, limits=cfg["eval"]["fixed_limits"])
        fl_curve_pts = [
            (
                summarize_policy_records(item["records"], seed=seed)["mean_steps"],
                summarize_policy_records(item["records"], seed=seed)["accuracy"],
            )
            for item in fl_sweep
        ]
        best_fl_recs = max(
            (item["records"] for item in fl_sweep),
            key=lambda recs: summarize_policy_records(recs, seed=seed)["accuracy"],
        )
        fl_by_task = {r["task_id"]: r["correct"] for r in best_fl_recs}

        hb_recs = evaluate_hindsight_best(base_test_labelled)
        hb_summ = summarize_policy_records(hb_recs, fixed_limit_curve=fl_curve_pts, seed=seed)

        # Reference lambda=0 correct tasks for computing right-answers-lost
        lam0_runs = runs_cache.get(("method_b", "lam_0.0", domain), [])
        lam0_correct_tasks = {r["task_id"] for r in lam0_runs if int(r.get("correct", 0)) == 1}

        b_step_pts: List[float] = []
        b_acc_pts: List[float] = []
        for lam in lambdas:
            setting_str = f"lam_{lam}"
            m_runs = runs_cache[("method_b", setting_str, domain)]
            summ, recs = evaluate_online_runs_against_hindsight(
                m_runs,
                reference_hindsight_map=ref_hindsight,
                fixed_limit_curve=fl_curve_pts,
                seed=seed,
            )
            # Compute right-answers-lost vs lambda=0
            cur_correct_tasks = {r["task_id"] for r in m_runs if int(r.get("correct", 0)) == 1}
            lost_count = len(lam0_correct_tasks - cur_correct_tasks)
            lost_rate = lost_count / max(1, len(m_runs))
            summ["right_answers_lost_count"] = lost_count
            summ["right_answers_lost_rate"] = round(lost_rate, 4)

            m_accs = [r["correct"] for r in recs if r["task_id"] in fl_by_task]
            fl_accs = [fl_by_task[r["task_id"]] for r in recs if r["task_id"] in fl_by_task]
            summ.update(paired_tests(m_accs, fl_accs, seed=seed))

            b_step_pts.append(summ["mean_steps"])
            b_acc_pts.append(summ["accuracy"])
            e4_rows.append(
                {
                    "domain": domain,
                    "method": "Method B (GRPO)",
                    "lambda": lam,
                    "setting": setting_str,
                    **summ,
                }
            )

        b_curve_auc = compute_auc_acc_steps(b_step_pts, b_acc_pts)
        for row in e4_rows:
            if row["domain"] == domain:
                row["lambda_sweep_auc"] = b_curve_auc

        # Evaluate Method C (E5) and compare to Hindsight Ceiling, Method A, and Method B
        for lam in method_c_lambdas:
            setting_str = f"lam_{lam}"
            c_runs = runs_cache[("method_c", setting_str, domain)]
            summ_c, recs_c = evaluate_online_runs_against_hindsight(
                c_runs,
                reference_hindsight_map=ref_hindsight,
                fixed_limit_curve=fl_curve_pts,
                seed=seed,
            )
            cur_correct_tasks = {r["task_id"] for r in c_runs if int(r.get("correct", 0)) == 1}
            lost_count = len(lam0_correct_tasks - cur_correct_tasks)
            summ_c["right_answers_lost_rate"] = round(lost_count / max(1, len(c_runs)), 4)
            summ_c["hindsight_ceiling_accuracy"] = hb_summ["accuracy"]
            summ_c["hindsight_ceiling_steps"] = hb_summ["mean_steps"]
            summ_c["accuracy_gap_to_ceiling"] = round(hb_summ["accuracy"] - summ_c["accuracy"], 4)
            summ_c["steps_gap_to_ceiling"] = round(summ_c["mean_steps"] - hb_summ["mean_steps"], 4)

            c_accs = [r["correct"] for r in recs_c if r["task_id"] in fl_by_task]
            fl_accs = [fl_by_task[r["task_id"]] for r in recs_c if r["task_id"] in fl_by_task]
            summ_c.update(paired_tests(c_accs, fl_accs, seed=seed))

            e5_rows.append(
                {
                    "domain": domain,
                    "method": "Method C (A+B)",
                    "lambda": lam,
                    "setting": setting_str,
                    **summ_c,
                }
            )

    pd.DataFrame(e4_rows).to_csv(results_dir / f"{prefix}e4_method_b_lambda_sweep.csv", index=False)
    pd.DataFrame(e5_rows).to_csv(results_dir / f"{prefix}e5_method_c.csv", index=False)

    out = {
        "smoke": smoke,
        "elapsed_sec": round(time.time() - t0, 2),
        "grpo_training": grpo_train_summaries,
        "e4_method_b": e4_rows,
        "e5_method_c": e5_rows,
    }
    with open(results_dir / f"{prefix}stage6_grpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Stage 6 complete. Saved E4 and E5 CSVs to {results_dir}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--grpo-steps", type=int, default=80)
    args = parser.parse_args()
    run_stage6(config_path=args.config, smoke=args.smoke, grpo_steps=args.grpo_steps)
