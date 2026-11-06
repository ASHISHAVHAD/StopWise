"""Bootstrap confidence intervals (1,000 resamples) and paired statistical tests."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute mean and 95% bootstrap confidence interval using 1,000 resamples."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "half_width": 0.0, "formatted": "0.000 ± 0.000"}
    mean_val = float(np.mean(arr))
    if arr.size == 1 or np.allclose(arr, arr[0]):
        return {
            "mean": mean_val,
            "ci_low": mean_val,
            "ci_high": mean_val,
            "half_width": 0.0,
            "formatted": f"{mean_val:.3f} ± 0.000",
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    hw = (high - low) / 2.0
    return {
        "mean": round(mean_val, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "half_width": round(hw, 4),
        "formatted": f"{mean_val:.3f} ± {hw:.3f}",
    }


def paired_tests(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Run paired bootstrap test and Wilcoxon signed-rank test on identical tasks (A - B)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"mean_diff": 0.0, "bootstrap_p": 1.0, "wilcoxon_p": 1.0}

    diffs = a - b
    mean_diff = float(np.mean(diffs))
    if np.allclose(diffs, 0.0):
        return {"mean_diff": 0.0, "bootstrap_p": 1.0, "wilcoxon_p": 1.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_diffs = np.mean(diffs[idx], axis=1)
    # Two-sided paired bootstrap p-value
    p_pos = float(np.mean(boot_diffs <= 0.0))
    p_neg = float(np.mean(boot_diffs >= 0.0))
    boot_p = min(1.0, 2.0 * min(p_pos, p_neg))

    try:
        _, wilcox_p = stats.wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
        wilcox_p = float(wilcox_p) if np.isfinite(wilcox_p) else 1.0
    except Exception:
        wilcox_p = 1.0

    return {
        "mean_diff": round(mean_diff, 4),
        "bootstrap_p": round(boot_p, 4),
        "wilcoxon_p": round(wilcox_p, 4),
    }

