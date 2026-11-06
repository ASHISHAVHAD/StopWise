"""Primary metrics: accuracy vs mean steps, normalized AUC, premature-stop, over-run, and steps saved."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.stats import bootstrap_ci


def compute_auc_acc_steps(
    mean_steps_list: Sequence[float],
    acc_list: Sequence[float],
    max_step_budget: float = 10.0,
) -> float:
    """Compute normalized Area Under the Accuracy-versus-Steps Curve over [1, max_step_budget].

    If a single operating point (s, a) is given, the step curve is modeled as reaching accuracy `a`
    at step `s` and maintaining `a` up to `max_step_budget` (so finishing earlier with high accuracy
    yields higher AUC).
    """
    if not mean_steps_list or not acc_list:
        return 0.0
    pts = sorted(zip([float(s) for s in mean_steps_list], [float(a) for a in acc_list]), key=lambda x: x[0])
    # Deduplicate x coordinates keeping max accuracy at identical step count
    dedup: Dict[float, float] = {}
    for x, y in pts:
        x_clamped = min(max_step_budget, max(1.0, x))
        dedup[x_clamped] = max(dedup.get(x_clamped, 0.0), y)
    xs = sorted(dedup.keys())
    ys = [dedup[x] for x in xs]

    if xs[0] > 1.0:
        xs = [1.0] + xs
        ys = [0.0] + ys
    if xs[-1] < max_step_budget:
        xs = xs + [max_step_budget]
        ys = ys + [ys[-1]]

    area = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
    norm_span = max(1e-6, max_step_budget - 1.0)
    return round(area / norm_span, 4)


def compute_steps_saved_at_equal_accuracy(
    method_acc: float,
    method_steps: float,
    fixed_limit_curve: Sequence[Tuple[float, float]],
) -> float:
    """Compute mean steps saved compared to the fixed-limit baseline at equal (or interpolated) accuracy.

    `fixed_limit_curve` is a sequence of `(mean_steps, accuracy)` pairs from fixed limits 1..10.
    If `method_acc` exceeds the best fixed limit accuracy, compares against the best fixed limit's steps.
    """
    if not fixed_limit_curve:
        return 0.0
    sorted_fl = sorted(fixed_limit_curve, key=lambda x: x[0])
    # Find earliest fixed limit that achieves >= method_acc - 1e-3
    for fl_steps, fl_acc in sorted_fl:
        if fl_acc >= method_acc - 1e-3:
            return round(float(fl_steps - method_steps), 4)
    # If method beats all fixed limits, compare against the highest-accuracy fixed limit
    best_fl_steps, _ = max(sorted_fl, key=lambda x: (x[1], -x[0]))
    return round(float(best_fl_steps - method_steps), 4)


def summarize_policy_records(
    records: Sequence[Dict[str, Any]],
    fixed_limit_curve: Sequence[Tuple[float, float]] = (),
    n_resamples: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate per-task evaluation records with 95% bootstrap confidence intervals."""
    accs = [float(r.get("correct", 0.0)) for r in records]
    steps = [float(r.get("steps", r.get("num_steps", 1.0))) for r in records]
    early = [float(r.get("premature_stop", 0.0)) for r in records]
    over = [float(r.get("over_run", 0.0)) for r in records]
    wasted = [float(r.get("wasted_steps", 0.0)) for r in records]

    acc_ci = bootstrap_ci(accs, n_resamples=n_resamples, seed=seed)
    steps_ci = bootstrap_ci(steps, n_resamples=n_resamples, seed=seed)
    early_ci = bootstrap_ci(early, n_resamples=n_resamples, seed=seed)
    over_ci = bootstrap_ci(over, n_resamples=n_resamples, seed=seed)
    wasted_ci = bootstrap_ci(wasted, n_resamples=n_resamples, seed=seed)

    auc = compute_auc_acc_steps([steps_ci["mean"]], [acc_ci["mean"]])
    saved = compute_steps_saved_at_equal_accuracy(acc_ci["mean"], steps_ci["mean"], fixed_limit_curve)

    return {
        "n": len(records),
        "accuracy": acc_ci["mean"],
        "accuracy_ci": acc_ci["formatted"],
        "accuracy_half_width": acc_ci["half_width"],
        "mean_steps": steps_ci["mean"],
        "mean_steps_ci": steps_ci["formatted"],
        "mean_steps_half_width": steps_ci["half_width"],
        "premature_stop_rate": early_ci["mean"],
        "premature_stop_ci": early_ci["formatted"],
        "over_run_rate": over_ci["mean"],
        "over_run_ci": over_ci["formatted"],
        "mean_wasted_steps": wasted_ci["mean"],
        "mean_wasted_ci": wasted_ci["formatted"],
        "auc_acc_vs_steps": auc,
        "steps_saved_vs_fixed": saved,
    }
