"""Stage 5: Train Method A (SFT on cut-off labels) and evaluate E3 on the held-out test sets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.agent.loop import LLMGenerator, run_agent_batch
from src.baselines.confidence_rule import sweep_confidence_thresholds
from src.baselines.fixed_limit import sweep_fixed_limits
from src.eval.evaluate import evaluate_online_runs_against_hindsight
from src.eval.metrics import compute_auc_acc_steps, summarize_policy_records
from src.eval.stats import paired_tests
from src.tools.retriever import HybridRetriever
from src.train.sft import build_sft_examples_from_runs, train_sft_model
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger, write_jsonl


def run_stage5(
    config_path: str = "configs/default.yaml",
    smoke: bool = False,
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

    logger = setup_logger("stage5", results_dir / "logs" / "stage5_sft_method_a.log")
    prefix = "smoke_" if smoke else ""
    if smoke:
        test_limit_per_domain = 10

    domains = cfg["datasets"]["domains"]
    all_train_examples: List[Dict[str, Any]] = []
    all_val_examples: List[Dict[str, Any]] = []

    for domain in domains:
        train_tasks = {t["task_id"]: t for t in read_jsonl(splits_dir / f"{domain}_train.jsonl")}
        val_tasks = {t["task_id"]: t for t in read_jsonl(splits_dir / f"{domain}_val.jsonl")}

        train_labelled = read_jsonl(runs_dir / f"{domain}_train_labelled.jsonl")
        val_labelled = read_jsonl(runs_dir / f"{domain}_val_labelled.jsonl")

        dom_train_ex = build_sft_examples_from_runs(
            train_labelled,
            train_tasks,
            balance_classes=True,
            include_readiness=True,
            seed=seed,
        )
        dom_val_ex = build_sft_examples_from_runs(
            val_labelled,
            val_tasks,
            balance_classes=False,
            include_readiness=True,
            seed=seed,
        )
        all_train_examples.extend(dom_train_ex)
        all_val_examples.extend(dom_val_ex)
        logger.info(f"Built SFT examples for {domain}: train={len(dom_train_ex)}, val={len(dom_val_ex)}")

    if not all_val_examples and len(all_train_examples) > 20:
        all_val_examples = all_train_examples[-20:]
        all_train_examples = all_train_examples[:-20]

    method_a_ckpt = ckpt_root / "method_a_sft"
    t0 = time.time()
    logger.info(f"Training Method A (SFT) on {len(all_train_examples)} examples -> {method_a_ckpt}...")
    sft_metrics = train_sft_model(
        train_examples=all_train_examples,
        val_examples=all_val_examples,
        output_dir=method_a_ckpt,
        base_model_name=cfg["model"]["base_model"],
        lora_r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        epochs=1 if smoke else int(cfg["sft"]["epochs"]),
        lr=float(cfg["sft"]["learning_rate"]),
        batch_size=int(cfg["sft"]["batch_size"]),
        grad_accum_steps=int(cfg["sft"]["grad_accum_steps"]),
        seed=seed,
    )
    logger.info(f"Method A SFT training metrics: {sft_metrics}")

    # Evaluate Method A online on the held-out test set per domain (E3)
    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_name=cfg["retriever"]["embedding_model"],
        reranker_model_name=cfg["retriever"]["reranker_model"],
        device="cuda",
    )
    retriever.build_or_load(force_rebuild=False)

    generator = LLMGenerator(
        model_name=cfg["model"]["base_model"],
        adapter_path=str(method_a_ckpt),
        device="cuda",
        dtype=cfg["model"]["dtype"],
    )

    e3_rows: List[Dict[str, Any]] = []
    e3_curve_rows: List[Dict[str, Any]] = []

    for domain in domains:
        test_tasks = read_jsonl(splits_dir / f"{domain}_test.jsonl")[:test_limit_per_domain]
        base_test_labelled = read_jsonl(runs_dir / f"{domain}_test_labelled.jsonl")

        # Build task-level reference hindsight map and fixed-limit baseline curve
        ref_hindsight: Dict[str, int] = {}
        for r in base_test_labelled:
            hs = r.get("hindsight", {}).get("first_ready_step")
            if hs is not None:
                tid = r["task_id"]
                ref_hindsight[tid] = min(ref_hindsight.get(tid, 99), int(hs))

        fl_sweep = sweep_fixed_limits(base_test_labelled, limits=cfg["eval"]["fixed_limits"])
        fl_curve_pts: List[Tuple[float, float]] = []
        best_fl_recs: List[Dict[str, Any]] = []
        best_fl_acc = -1.0
        for item in fl_sweep:
            s_fl = summarize_policy_records(item["records"], seed=seed)
            fl_curve_pts.append((s_fl["mean_steps"], s_fl["accuracy"]))
            if s_fl["accuracy"] > best_fl_acc:
                best_fl_acc = s_fl["accuracy"]
                best_fl_recs = item["records"]

        out_runs_file = runs_dir / f"{domain}_test_method_a_runs.jsonl"
        if out_runs_file.exists():
            ma_runs = read_jsonl(out_runs_file)
        else:
            logger.info(f"Running Method A online evaluation on {domain} ({len(test_tasks)} test tasks)...")
            ma_runs = run_agent_batch(
                tasks=test_tasks,
                generator=generator,
                retriever=retriever,
                max_steps=7,
                temperature=0.2,
                few_shot=True,
                include_readiness=True,
                run_id_prefix="method_a",
                sample_idx=0,
                batch_size=32,
            )
            write_jsonl(out_runs_file, ma_runs)

        summary_ma, ma_recs = evaluate_online_runs_against_hindsight(
            ma_runs,
            reference_hindsight_map=ref_hindsight,
            fixed_limit_curve=fl_curve_pts,
            seed=seed,
        )

        # Match tasks for paired test against best fixed limit (first sample per task)
        fl_by_task = {r["task_id"]: r["correct"] for r in best_fl_recs}
        ma_accs = [r["correct"] for r in ma_recs if r["task_id"] in fl_by_task]
        fl_accs = [fl_by_task[r["task_id"]] for r in ma_recs if r["task_id"] in fl_by_task]
        summary_ma.update(paired_tests(ma_accs, fl_accs, seed=seed))

        # Also trace Method A curve across readiness thresholds
        # Attach cutoffs from base_test_labelled by task_id for sub-trajectory threshold evaluation
        cutoff_by_task = {r["task_id"]: r.get("cutoff_labels", []) for r in base_test_labelled}
        ma_enriched: List[Dict[str, Any]] = []
        for r in ma_runs:
            er = dict(r)
            er["cutoff_labels"] = cutoff_by_task.get(r["task_id"], [])
            er["hindsight"] = {"first_ready_step": ref_hindsight.get(r["task_id"])}
            ma_enriched.append(er)

        ma_sweep = sweep_confidence_thresholds(
            ma_enriched,
            thresholds=cfg["eval"]["confidence_thresholds"],
            score_key="readiness",
        )
        c_steps: List[float] = [summary_ma["mean_steps"]]
        c_accs: List[float] = [summary_ma["accuracy"]]
        for item in ma_sweep:
            s_tau = summarize_policy_records(item["records"], fixed_limit_curve=fl_curve_pts, seed=seed)
            c_steps.append(s_tau["mean_steps"])
            c_accs.append(s_tau["accuracy"])
            e3_curve_rows.append(
                {
                    "domain": domain,
                    "method": "Method A (SFT)",
                    "setting": f"tau_{item['threshold']}",
                    "param": item["threshold"],
                    **s_tau,
                }
            )
        summary_ma["auc_acc_vs_steps"] = compute_auc_acc_steps(c_steps, c_accs)
        e3_curve_rows.append(
            {
                "domain": domain,
                "method": "Method A (SFT)",
                "setting": "policy_stop",
                "param": 1.0,
                **summary_ma,
            }
        )
        e3_rows.append({"domain": domain, "method": "Method A (SFT)", **summary_ma})
        logger.info(f"E3 Method A ({domain}): acc={summary_ma['accuracy_ci']}, steps={summary_ma['mean_steps_ci']}")

    pd.DataFrame(e3_rows).to_csv(results_dir / f"{prefix}e3_method_a.csv", index=False)
    pd.DataFrame(e3_curve_rows).to_csv(results_dir / f"{prefix}e3_method_a_curves.csv", index=False)

    out = {
        "smoke": smoke,
        "elapsed_sec": round(time.time() - t0, 2),
        "sft_validation_metrics": sft_metrics,
        "e3_method_a": e3_rows,
    }
    with open(results_dir / f"{prefix}stage5_sft_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_stage5(config_path=args.config, smoke=args.smoke)
