"""Stage 4: Run E1 (Problem Size Measurement) and E2 (All Baselines) with 95% bootstrap CIs and paired tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.baselines.confidence_rule import sweep_confidence_thresholds
from src.baselines.fixed_limit import sweep_fixed_limits
from src.baselines.hindsight_best import evaluate_hindsight_best
from src.baselines.own_decision import evaluate_own_decision
from src.eval.metrics import compute_auc_acc_steps, summarize_policy_records
from src.eval.stats import bootstrap_ci, paired_tests
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger


def run_stage4(config_path: str = "configs/default.yaml", smoke: bool = False) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["primary_seed"])
    set_seed(seed)

    runs_dir = Path(cfg["project"]["runs_dir"]) / ("smoke" if smoke else "full")
    results_dir = Path(cfg["project"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage4", results_dir / "logs" / "stage4_baselines_e1_e2.log")

    prefix = "smoke_" if smoke else ""
    domains = cfg["datasets"]["domains"]

    e1_rows: List[Dict[str, Any]] = []
    e2_curve_rows: List[Dict[str, Any]] = []
    e2_summary_rows: List[Dict[str, Any]] = []

    for domain in domains:
        test_file = runs_dir / f"{domain}_test_labelled.jsonl"
        if not test_file.exists():
            continue
        test_runs = read_jsonl(test_file)
        if not test_runs:
            continue

        # --- E1: Problem size on untrained agent ---
        cont_after_ready = [float(r["hindsight"]["continued_after_ready"]) for r in test_runs]
        wasted = [float(r["hindsight"]["wasted_steps"]) for r in test_runs]
        early_stop = [float(r["hindsight"]["stopped_too_early"]) for r in test_runs]
        own_steps = [float(r["num_steps"]) for r in test_runs]
        own_acc = [float(r["correct"]) for r in test_runs]
        hindsight_acc = [float(r["hindsight"]["hindsight_correct"]) for r in test_runs]

        c_ci = bootstrap_ci(cont_after_ready, seed=seed)
        w_ci = bootstrap_ci(wasted, seed=seed)
        e_ci = bootstrap_ci(early_stop, seed=seed)
        s_ci = bootstrap_ci(own_steps, seed=seed)
        a_ci = bootstrap_ci(own_acc, seed=seed)
        h_ci = bootstrap_ci(hindsight_acc, seed=seed)

        e1_rows.append(
            {
                "domain": domain,
                "n_runs": len(test_runs),
                "untrained_accuracy": a_ci["mean"],
                "untrained_accuracy_ci": a_ci["formatted"],
                "hindsight_ceiling_accuracy": h_ci["mean"],
                "hindsight_ceiling_ci": h_ci["formatted"],
                "mean_steps": s_ci["mean"],
                "mean_steps_ci": s_ci["formatted"],
                "share_continuing_after_ready": c_ci["mean"],
                "share_continuing_after_ready_ci": c_ci["formatted"],
                "mean_wasted_steps": w_ci["mean"],
                "mean_wasted_steps_ci": w_ci["formatted"],
                "share_stopping_too_early": e_ci["mean"],
                "share_stopping_too_early_ci": e_ci["formatted"],
            }
        )

        # --- E2: Baselines ---
        # 1. Fixed step limits (1..10)
        fl_sweep = sweep_fixed_limits(test_runs, limits=cfg["eval"]["fixed_limits"])
        fl_curve_pts: List[Tuple[float, float]] = []
        fl_step_means: List[float] = []
        fl_acc_means: List[float] = []
        best_fl_summary: Dict[str, Any] = {}
        best_fl_records: List[Dict[str, Any]] = []

        for item in fl_sweep:
            lim = item["limit"]
            recs = item["records"]
            summ = summarize_policy_records(recs, seed=seed)
            fl_curve_pts.append((summ["mean_steps"], summ["accuracy"]))
            fl_step_means.append(summ["mean_steps"])
            fl_acc_means.append(summ["accuracy"])
            e2_curve_rows.append(
                {
                    "domain": domain,
                    "method": "fixed_limit",
                    "setting": f"limit_{lim}",
                    "param": lim,
                    **summ,
                }
            )
            if not best_fl_summary or summ["accuracy"] > best_fl_summary["accuracy"]:
                best_fl_summary = dict(summ)
                best_fl_summary["setting"] = f"limit_{lim}"
                best_fl_records = recs

        fl_curve_auc = compute_auc_acc_steps(fl_step_means, fl_acc_means)
        best_fl_summary["auc_acc_vs_steps"] = fl_curve_auc
        e2_summary_rows.append(
            {
                "domain": domain,
                "method": "Fixed Limit (Best)",
                "setting": best_fl_summary.get("setting", "limit_7"),
                **best_fl_summary,
            }
        )

        # 2. Untrained Own Decision
        own_recs = evaluate_own_decision(test_runs)
        own_summ = summarize_policy_records(own_recs, fixed_limit_curve=fl_curve_pts, seed=seed)
        ptest_own = paired_tests(
            [r["correct"] for r in own_recs],
            [r["correct"] for r in best_fl_records],
            seed=seed,
        )
        own_summ.update(ptest_own)
        e2_curve_rows.append({"domain": domain, "method": "own_decision", "setting": "default", "param": 0, **own_summ})
        e2_summary_rows.append({"domain": domain, "method": "Own Decision (Untrained)", "setting": "default", **own_summ})

        # 3. Confidence Rule (sweep readiness thresholds and token_confidence thresholds)
        for score_key, label_name in [("readiness", "confidence_readiness"), ("token_confidence", "confidence_prob")]:
            conf_sweep = sweep_confidence_thresholds(
                test_runs,
                thresholds=cfg["eval"]["confidence_thresholds"],
                score_key=score_key,
            )
            c_steps: List[float] = []
            c_accs: List[float] = []
            best_c_summ: Dict[str, Any] = {}
            best_c_recs: List[Dict[str, Any]] = []
            for item in conf_sweep:
                tau = item["threshold"]
                recs = item["records"]
                summ = summarize_policy_records(recs, fixed_limit_curve=fl_curve_pts, seed=seed)
                c_steps.append(summ["mean_steps"])
                c_accs.append(summ["accuracy"])
                e2_curve_rows.append(
                    {
                        "domain": domain,
                        "method": label_name,
                        "setting": f"tau_{tau}",
                        "param": tau,
                        **summ,
                    }
                )
                if not best_c_summ or (summ["accuracy"] - 0.02 * summ["mean_steps"]) > (
                    best_c_summ["accuracy"] - 0.02 * best_c_summ["mean_steps"]
                ):
                    best_c_summ = dict(summ)
                    best_c_summ["setting"] = f"tau_{tau}"
                    best_c_recs = recs

            best_c_summ["auc_acc_vs_steps"] = compute_auc_acc_steps(c_steps, c_accs)
            best_c_summ.update(
                paired_tests(
                    [r["correct"] for r in best_c_recs],
                    [r["correct"] for r in best_fl_records],
                    seed=seed,
                )
            )
            e2_summary_rows.append(
                {
                    "domain": domain,
                    "method": f"Confidence Rule ({score_key})",
                    "setting": best_c_summ.get("setting", "tau_0.8"),
                    **best_c_summ,
                }
            )

        # 4. Hindsight Best (Oracle ceiling)
        hb_recs = evaluate_hindsight_best(test_runs)
        hb_summ = summarize_policy_records(hb_recs, fixed_limit_curve=fl_curve_pts, seed=seed)
        hb_summ.update(
            paired_tests(
                [r["correct"] for r in hb_recs],
                [r["correct"] for r in best_fl_records],
                seed=seed,
            )
        )
        e2_curve_rows.append({"domain": domain, "method": "hindsight_best", "setting": "oracle", "param": 1.0, **hb_summ})
        e2_summary_rows.append({"domain": domain, "method": "Hindsight Best (Oracle)", "setting": "oracle", **hb_summ})

    df_e1 = pd.DataFrame(e1_rows)
    df_e2_curves = pd.DataFrame(e2_curve_rows)
    df_e2_summary = pd.DataFrame(e2_summary_rows)

    df_e1.to_csv(results_dir / f"{prefix}e1_problem_size.csv", index=False)
    df_e2_curves.to_csv(results_dir / f"{prefix}e2_baselines_curves.csv", index=False)
    df_e2_summary.to_csv(results_dir / f"{prefix}e2_baselines_summary.csv", index=False)

    out = {
        "e1_problem_size": e1_rows,
        "e2_baselines_summary": e2_summary_rows,
    }
    with open(results_dir / f"{prefix}stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Stage 4 complete. Saved E1 and E2 CSVs to {results_dir}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_stage4(config_path=args.config, smoke=args.smoke)

