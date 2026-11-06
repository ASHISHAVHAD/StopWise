"""Regenerate all report figures strictly from CSV tables in `results/`."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
    }
)


def make_all_figures(results_dir: str | Path = "results", prefix: str = "") -> None:
    res = Path(results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # 1. Stage 3: Ready fraction by step index
    s3_csv = res / f"{prefix}stage3_ready_by_step.csv"
    if not s3_csv.exists() and prefix == "":
        s3_csv = res / "stage3_ready_by_step.csv"
    if s3_csv.exists():
        df_s3 = pd.read_csv(s3_csv)
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for dom, grp in df_s3.groupby("domain"):
            grp = grp.sort_values("step_index")
            ax.plot(grp["step_index"], grp["mean_ready_fraction"], marker="o", linewidth=2, label=dom)
        ax.axhline(0.6, color="crimson", linestyle="--", linewidth=1.2, label="Ready Threshold (0.6)")
        ax.set_xlabel("Trajectory Step Index (t)")
        ax.set_ylabel("Mean Forced-Answer Ready Fraction")
        ax.set_title("Cut-Off Test Readiness Progression by Step Index")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_stage3_ready_by_step.png")
        fig.savefig(res / f"{prefix}stage3_ready_by_step.png")
        plt.close(fig)

    # 2. E1: Problem size bar chart
    e1_csv = res / f"{prefix}e1_problem_size.csv"
    if e1_csv.exists():
        df_e1 = pd.read_csv(e1_csv)
        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(len(df_e1))
        w = 0.26
        ax1.bar(x - w, df_e1["share_continuing_after_ready"], width=w, label="Share Continuing After Ready", color="#d95f02")
        ax1.bar(x, df_e1["share_stopping_too_early"], width=w, label="Share Stopping Too Early", color="#7570b3")
        ax1.bar(x + w, df_e1["mean_wasted_steps"] / 5.0, width=w, label="Mean Wasted Steps (scaled /5)", color="#1b9e77")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_e1["domain"])
        ax1.set_ylabel("Rate / Scaled Steps")
        ax1.set_title("E1: Untrained Agent Over-Thinking & Premature Stopping by Domain")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e1_problem_size.png")
        plt.close(fig)

    # 3. E2–E5: Accuracy vs Mean Steps Curves per Domain
    e2_csv = res / f"{prefix}e2_baselines_curves.csv"
    e3_csv = res / f"{prefix}e3_method_a.csv"
    e4_csv = res / f"{prefix}e4_method_b_lambda_sweep.csv"
    e5_csv = res / f"{prefix}e5_method_c.csv"
    if e2_csv.exists():
        df_e2 = pd.read_csv(e2_csv)
        df_e3 = pd.read_csv(e3_csv) if e3_csv.exists() else pd.DataFrame()
        df_e4 = pd.read_csv(e4_csv) if e4_csv.exists() else pd.DataFrame()
        df_e5 = pd.read_csv(e5_csv) if e5_csv.exists() else pd.DataFrame()

        domains = list(df_e2["domain"].unique())
        fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 4.4), sharey=False)
        if len(domains) == 1:
            axes = [axes]

        for ax, dom in zip(axes, domains):
            sub2 = df_e2[df_e2["domain"] == dom]
            fl = sub2[sub2["method"] == "fixed_limit"].sort_values("param")
            if not fl.empty:
                ax.plot(fl["mean_steps"], fl["accuracy"], "-o", color="#377eb8", linewidth=1.8, markersize=4, label="Fixed Limit (1..10)")

            cr = sub2[sub2["method"] == "confidence_readiness"].sort_values("mean_steps")
            if not cr.empty:
                ax.plot(cr["mean_steps"], cr["accuracy"], "--^", color="#984ea3", linewidth=1.4, markersize=4, label="Confidence Rule")

            own = sub2[sub2["method"] == "own_decision"]
            if not own.empty:
                ax.scatter(own["mean_steps"], own["accuracy"], color="#e41a1c", s=70, marker="X", zorder=5, label="Own Decision")

            hb = sub2[sub2["method"] == "hindsight_best"]
            if not hb.empty:
                ax.scatter(hb["mean_steps"], hb["accuracy"], color="gold", edgecolor="black", s=110, marker="*", zorder=6, label="Hindsight Best (Ceiling)")

            if not df_e3.empty:
                sub3 = df_e3[df_e3["domain"] == dom]
                if not sub3.empty:
                    ax.scatter(sub3["mean_steps"], sub3["accuracy"], color="#4daf4a", s=80, marker="s", zorder=6, label="Method A (SFT)")

            if not df_e4.empty:
                sub4 = df_e4[df_e4["domain"] == dom].sort_values("mean_steps")
                if not sub4.empty:
                    ax.plot(sub4["mean_steps"], sub4["accuracy"], "-D", color="#ff7f00", linewidth=1.8, markersize=5, label="Method B (GRPO λ sweep)")

            if not df_e5.empty:
                sub5 = df_e5[df_e5["domain"] == dom].sort_values("mean_steps")
                if not sub5.empty:
                    ax.scatter(sub5["mean_steps"], sub5["accuracy"], color="#a65628", edgecolor="black", s=95, marker="P", zorder=7, label="Method C (SFT+GRPO)")

            ax.set_title(f"Domain: {dom}")
            ax.set_xlabel("Mean Steps Used")
            ax.set_ylabel("Task Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=7.5)

        fig.suptitle("Accuracy vs. Mean Steps Across Domains (E2–E5)", y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e2_e5_accuracy_vs_steps.png", bbox_inches="tight")
        plt.close(fig)

    # 4. Early-Stop vs Over-Run Bar Chart (E2, E3, E4, E5)
    e2_sum_csv = res / f"{prefix}e2_baselines_summary.csv"
    if e2_sum_csv.exists() and e3_csv.exists() and e4_csv.exists() and e5_csv.exists():
        b_df = pd.read_csv(e2_sum_csv)
        a_df = pd.read_csv(e3_csv)
        b_grpo = pd.read_csv(e4_csv)
        c_df = pd.read_csv(e5_csv)

        b_01 = b_grpo[b_grpo["lambda"] == 0.1] if "lambda" in b_grpo.columns else b_grpo
        c_01 = c_df[c_df["lambda"] == 0.1] if "lambda" in c_df.columns else c_df

        combined = pd.concat([b_df, a_df, b_01, c_01], ignore_index=True)
        agg = (
            combined.groupby("method", as_index=False)
            .agg(
                premature_stop_rate=("premature_stop_rate", "mean"),
                over_run_rate=("over_run_rate", "mean"),
            )
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.3))
        x = np.arange(len(agg))
        w = 0.35
        ax.bar(x - w / 2, agg["premature_stop_rate"], width=w, label="Premature-Stop Rate", color="#e41a1c")
        ax.bar(x + w / 2, agg["over_run_rate"], width=w, label="Over-Run Rate (Continued After Ready)", color="#377eb8")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["method"], rotation=20, ha="right")
        ax.set_ylabel("Rate Across Test Tasks")
        ax.set_title("Premature Stopping vs. Over-Running Across Methods (Macro-Average)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e3_e5_early_overrun_bars.png")
        plt.close(fig)

    # 5. E4: Reward Hacking & Lambda Sweep
    if e4_csv.exists():
        df_e4 = pd.read_csv(e4_csv)
        lam_agg = (
            df_e4.groupby("lambda", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                mean_steps=("mean_steps", "mean"),
                early_answer_rate=("early_answer_rate", "mean"),
                right_answers_lost_rate=("right_answers_lost_rate", "mean"),
            )
            .sort_values("lambda")
        )
        fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
        ax1.plot(lam_agg["lambda"], lam_agg["accuracy"], "-o", linewidth=2, label="Accuracy", color="#1b9e77")
        ax1.plot(lam_agg["lambda"], lam_agg["early_answer_rate"], "--s", linewidth=2, label="Early-Answer (Step 1) Rate", color="#d95f02")
        ax1.plot(lam_agg["lambda"], lam_agg["right_answers_lost_rate"], ":^", linewidth=2, label="Right-Answers-Lost vs λ=0", color="#e7298a")
        ax1.set_xlabel("Effort Penalty Weight (λ)")
        ax1.set_ylabel("Rate / Probability")
        ax1.set_ylim(-0.02, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(lam_agg["lambda"], lam_agg["mean_steps"], "-.d", linewidth=2, color="#7570b3", label="Mean Steps Used")
        ax2.set_ylabel("Mean Steps")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        ax1.set_title("E4: Method B (GRPO) Sensitivity to Effort Penalty λ & Reward Hacking Monitor")
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e4_reward_hacking.png")
        plt.close(fig)

    # 6. E6: Cross-Domain Transfer
    e6_csv = res / f"{prefix}e6_transfer.csv"
    if e6_csv.exists():
        df_e6 = pd.read_csv(e6_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
        x = np.arange(len(df_e6))
        w = 0.26
        ax1.bar(x - w, df_e6["untrained_own_accuracy"], width=w, label="Untrained Own Decision", color="#999999")
        ax1.bar(x, df_e6["accuracy"], width=w, label="Transfer SFT (Trained on Other 2 Domains)", color="#ff7f00")
        ax1.bar(x + w, df_e6["indomain_sft_accuracy"], width=w, label="In-Domain SFT (Method A)", color="#4daf4a")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_e6["test_domain"])
        ax1.set_ylabel("Accuracy")
        ax1.set_title("E6 Transfer: Accuracy on Held-Out Domain")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend(fontsize=8)

        ax2.bar(x - w, df_e6["untrained_own_steps"], width=w, label="Untrained Own Decision", color="#999999")
        ax2.bar(x, df_e6["mean_steps"], width=w, label="Transfer SFT", color="#ff7f00")
        ax2.bar(x + w, df_e6["indomain_sft_steps"], width=w, label="In-Domain SFT", color="#4daf4a")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_e6["test_domain"])
        ax2.set_ylabel("Mean Steps")
        ax2.set_title("E6 Transfer: Mean Steps on Held-Out Domain")
        ax2.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e6_transfer.png")
        plt.close(fig)

    # 7. E7: Readiness Calibration Plot & Threshold Sweep
    e7_bins_csv = res / f"{prefix}e7_calibration_bins.csv"
    e7_th_csv = res / f"{prefix}e7_threshold_sweep.csv"
    if e7_bins_csv.exists() and e7_th_csv.exists():
        df_bins = pd.read_csv(e7_bins_csv)
        df_th = pd.read_csv(e7_th_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
        for m_name, grp in df_bins.groupby("model"):
            non_empty = grp[grp["count"] > 0].sort_values("bin_center")
            if not non_empty.empty:
                ax1.plot(non_empty["mean_predicted"], non_empty["empirical_accuracy"], "-o", linewidth=1.8, label=m_name)
        ax1.set_xlabel("Predicted Readiness Score")
        ax1.set_ylabel("Empirical Ready / Correct Rate")
        ax1.set_title("E7: Readiness Score Reliability Diagram (10 Bins)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        df_th = df_th.sort_values("threshold")
        ax2.plot(df_th["threshold"], df_th["premature_stop_rate"], "-s", color="#e41a1c", linewidth=2, label="Premature-Stop Rate")
        ax2.plot(df_th["threshold"], df_th["over_run_rate"], "-^", color="#377eb8", linewidth=2, label="Over-Run Rate")
        ax2.plot(df_th["threshold"], df_th["accuracy"], "-o", color="#4daf4a", linewidth=2, label="Accuracy")
        ax2.set_xlabel("Readiness Stopping Threshold (τ)")
        ax2.set_ylabel("Rate")
        ax2.set_title("E7: Stopping Error Rates vs. Readiness Threshold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e7_calibration.png")
        plt.close(fig)

    # 8. E8: Ablations Chart
    e8_csv = res / f"{prefix}e8_ablations.csv"
    if e8_csv.exists():
        df_e8 = pd.read_csv(e8_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
        y = np.arange(len(df_e8))
        ax1.barh(y, df_e8["accuracy"], color="#377eb8")
        ax1.set_yticks(y)
        ax1.set_yticklabels(df_e8["ablation"])
        ax1.set_xlabel("Accuracy")
        ax1.set_title("E8 Ablations: Task Accuracy")
        ax1.grid(True, axis="x", alpha=0.3)

        ax2.barh(y, df_e8["mean_steps"], color="#ff7f00")
        ax2.set_xlabel("Mean Steps Used")
        ax2.set_title("E8 Ablations: Mean Steps")
        ax2.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e8_ablations.png")
        plt.close(fig)

    # 9. E9: Retrieval Quality Adaptation
    e9_csv = res / f"{prefix}e9_retrieval_quality.csv"
    if e9_csv.exists():
        df_e9 = pd.read_csv(e9_csv)
        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(len(df_e9))
        w = 0.35
        ax1.bar(x - w / 2, df_e9["accuracy"], width=w, color="#4daf4a", label="Accuracy")
        ax1.set_ylabel("Search QA Accuracy")
        ax1.set_ylim(0, 1.0)
        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, df_e9["mean_steps"], width=w, color="#377eb8", label="Mean Steps Taken")
        ax2.set_ylabel("Mean Steps")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_e9["retrieval_setting"], rotation=15, ha="right")
        ax1.set_title("E9: Learned Policy Adaptation to Retrieval Quality (Search QA)")
        ax1.grid(True, axis="y", alpha=0.3)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e9_retrieval_quality.png")
        plt.close(fig)

    # 10. E10: Failure Taxonomy Breakdown
    e10_csv = res / f"{prefix}e10_failure_taxonomy.csv"
    if e10_csv.exists():
        df_e10 = pd.read_csv(e10_csv)
        cats = [
            "repeated_search",
            "reopened_source",
            "repeated_check",
            "repeated_calculation",
            "stopped_too_early",
            "looped_until_limit",
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        x = np.arange(len(df_e10))
        bottom = np.zeros(len(df_e10))
        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
        for cat, col in zip(cats, colors):
            col_name = f"{cat}_rate_overall"
            if col_name in df_e10.columns:
                vals = df_e10[col_name].values
                ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " ").title(), color=col)
                bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(df_e10["method"], rotation=18, ha="right")
        ax.set_ylabel("Overall Error Rate by Failure Mode")
        ax.set_title("E10: Failure Mode Taxonomy Across Stopping Methods")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(res / f"{prefix}fig_e10_failure_taxonomy.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()
    make_all_figures(results_dir=args.results_dir, prefix=args.prefix)

