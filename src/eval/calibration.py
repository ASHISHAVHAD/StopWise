"""Readiness score calibration (E7): reliability diagram bins, ECE, Brier score, and threshold sweep."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def compute_calibration_metrics(
    predicted_readiness: Sequence[float],
    empirical_outcomes: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute 10-bin reliability curve, Expected Calibration Error (ECE), and Brier score."""
    preds = np.clip(np.asarray(predicted_readiness, dtype=np.float64), 0.0, 1.0)
    trues = np.clip(np.asarray(empirical_outcomes, dtype=np.float64), 0.0, 1.0)

    if preds.size == 0:
        return {"ece": 0.0, "brier": 0.0, "bins": []}

    brier = float(np.mean((preds - trues) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data: List[Dict[str, Any]] = []
    ece = 0.0
    n_total = float(preds.size)

    for b in range(n_bins):
        low, high = float(bin_edges[b]), float(bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (preds >= low) & (preds <= high)
        else:
            mask = (preds >= low) & (preds < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(preds[mask]))
            mean_true = float(np.mean(trues[mask]))
            gap = abs(mean_pred - mean_true)
            ece += (count / n_total) * gap
        else:
            mean_pred = (low + high) / 2.0
            mean_true = 0.0
            gap = 0.0

        bins_data.append(
            {
                "bin_index": b,
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "bin_center": round((low + high) / 2.0, 3),
                "count": count,
                "mean_predicted": round(mean_pred, 4),
                "empirical_accuracy": round(mean_true, 4),
                "abs_gap": round(gap, 4),
            }
        )

    return {
        "n_samples": int(preds.size),
        "ece": round(float(ece), 4),
        "brier": round(brier, 4),
        "bins": bins_data,
    }

