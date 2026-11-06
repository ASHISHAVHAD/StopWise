"""Stage 3: Run cut-off labelling across saved runs, compute hindsight stop steps, and plot ready fraction by step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agent.loop import LLMGenerator
from src.labels.cutoff_labels import label_runs_with_cutoffs
from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl, setup_logger, write_jsonl


def run_stage3(
    config_path: str = "configs/default.yaml",
    smoke: bool = False,
    forced_samples: int = 5,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["primary_seed"])
    set_seed(seed)

    data_dir = Path(cfg["project"]["data_dir"])
    splits_dir = data_dir / ("splits_smoke" if smoke else "splits")
    runs_dir = Path(cfg["project"]["runs_dir"]) / ("smoke" if smoke else "full")
    results_dir = Path(cfg["project"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("stage3", results_dir / "logs" / "stage3_cutoff_labels.log")
    summary_file = results_dir / ("stage3_smoke_summary.json" if smoke else "stage3_cutoff_summary.json")

    if smoke:
        forced_samples = 3

    generator = LLMGenerator(
        model_name=cfg["model"]["base_model"],
        device="cuda",
        dtype=cfg["model"]["dtype"],
    )

    domains = cfg["datasets"]["domains"]
    domain_stats: Dict[str, Any] = {}
    step_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for split_kind in ["train", "val", "test"]:
        for domain in domains:
            tasks = read_jsonl(splits_dir / f"{domain}_{split_kind}.jsonl")
            task_map = {t["task_id"]: t for t in tasks}

            in_file = runs_dir / f"{domain}_{split_kind}_runs.jsonl"
            out_file = runs_dir / f"{domain}_{split_kind}_labelled.jsonl"
            if not in_file.exists():
                continue

            raw_runs = read_jsonl(in_file)
            existing_labelled = read_jsonl(out_file)
            done_ids = {r["run_id"] for r in existing_labelled}
            pending_runs = [r for r in raw_runs if r["run_id"] not in done_ids]

            if pending_runs:
                logger.info(
                    f"Labelling {len(pending_runs)} runs for {domain} ({split_kind}) "
                    f"with {forced_samples} forced samples per step..."
                )
                # Process in chunks of 40 runs so progress is saved incrementally
                chunk_size = 40
                for start in range(0, len(pending_runs), chunk_size):
                    sub_runs = pending_runs[start : start + chunk_size]
                    labelled_sub = label_runs_with_cutoffs(
                        runs=sub_runs,
                        task_map=task_map,
                        generator=generator,
                        forced_samples=forced_samples,
                        ready_threshold=float(cfg["cutoff"]["ready_threshold"]),
                        temperature=float(cfg["cutoff"]["temperature"]),
                        batch_size=48,
                    )
                    write_jsonl(out_file, labelled_sub, append=True)
                    existing_labelled.extend(labelled_sub)

            # Aggregate statistics
            n_ready = 0
            n_not_ready = 0
            wasted_list: List[int] = []
            for r in existing_labelled:
                h = r.get("hindsight", {})
                wasted_list.append(int(h.get("wasted_steps", 0)))
                for c in r.get("cutoff_labels", []):
                    if c.get("label") == "ready":
                        n_ready += 1
                    else:
                        n_not_ready += 1
                    step_rows.append(
                        {
                            "domain": domain,
                            "split": split_kind,
                            "run_id": r["run_id"],
                            "step_index": int(c["step_index"]),
                            "ready_fraction": float(c["ready_fraction"]),
                            "ready_int": int(c["ready_int"]),
                        }
                    )
                # Also record final ANSWER step readiness (= final correctness) for step curve completeness
                steps = r.get("steps", [])
                if steps and steps[-1].get("action") == "ANSWER":
                    step_rows.append(
                        {
                            "domain": domain,
                            "split": split_kind,
                            "run_id": r["run_id"],
                            "step_index": int(steps[-1]["step_index"]),
                            "ready_fraction": float(r["correct"]),
                            "ready_int": int(r["correct"]),
                        }
                    )

            key = f"{domain}_{split_kind}"
            domain_stats[key] = {
                "num_runs": len(existing_labelled),
                "ready_cutoffs": n_ready,
                "not_ready_cutoffs": n_not_ready,
                "total_cutoffs": n_ready + n_not_ready,
                "mean_wasted_steps": round(sum(wasted_list) / max(1, len(wasted_list)), 4),
            }
            logger.info(f"Stage 3 stats for {key}: {domain_stats[key]}")

    # Save CSV and plot of ready_fraction by step index
    df_steps = pd.DataFrame(step_rows)
    if not df_steps.empty:
        agg_df = (
            df_steps.groupby(["domain", "step_index"], as_index=False)
            .agg(
                mean_ready_fraction=("ready_fraction", "mean"),
                share_ready=("ready_int", "mean"),
                count=("ready_fraction", "count"),
            )
        )
        csv_path = results_dir / ("stage3_smoke_ready_by_step.csv" if smoke else "stage3_ready_by_step.csv")
        agg_df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
        for dom, grp in agg_df.groupby("domain"):
            ax.plot(grp["step_index"], grp["mean_ready_fraction"], marker="o", linewidth=2, label=dom)
        ax.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="Ready threshold (0.6)")
        ax.set_xlabel("Step Index (t)")
        ax.set_ylabel("Mean Ready Fraction (5 Forced Samples)")
        ax.set_title("Cut-Off Test Readiness Fraction by Step Index Across Domains")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig_path = results_dir / ("stage3_smoke_ready_by_step.png" if smoke else "stage3_ready_by_step.png")
        fig.savefig(fig_path)
        plt.close(fig)

    summary = {
        "smoke": smoke,
        "forced_samples": forced_samples,
        "elapsed_sec": round(time.time() - t0, 2),
        "splits": domain_stats,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Stage 3 finished. Summary written to {summary_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--forced-samples", type=int, default=5)
    args = parser.parse_args()
    run_stage3(config_path=args.config, smoke=args.smoke, forced_samples=args.forced_samples)

