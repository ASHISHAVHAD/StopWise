"""Fixed step-limit stopping baseline (limits 1 to 10)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def evaluate_fixed_limit_on_run(run: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Evaluate a labelled run under a fixed step limit `limit` in {1..10}.

    If the run naturally stopped at or before `limit`, it takes `run['num_steps']` steps and gets `run['correct']`.
    If `limit < run['num_steps']`, the run is cut off at step `limit` and uses the forced-answer accuracy at step `limit`.
    """
    n_steps = int(run.get("num_steps", len(run.get("steps", []))))
    cutoffs = {int(c["step_index"]): c for c in run.get("cutoff_labels", [])}
    hindsight = run.get("hindsight", {})
    first_ready = hindsight.get("first_ready_step")

    if n_steps <= limit:
        used_steps = n_steps
        correct = float(run.get("correct", 0))
    else:
        used_steps = limit
        if limit in cutoffs:
            # Expected accuracy when forced to answer at `limit` (or majority vote of forced samples)
            c_info = cutoffs[limit]
            sample_corr = c_info.get("sample_correct", [])
            if sample_corr:
                # First forced sample (deterministic single rollout equivalent) or ready_fraction
                correct = float(c_info.get("ready_fraction", sample_corr[0]))
            else:
                correct = float(c_info.get("ready_fraction", 0.0))
        else:
            correct = 0.0

    premature_stop = int(first_ready is not None and used_steps < int(first_ready))
    over_run = int(first_ready is not None and used_steps > int(first_ready))
    wasted = max(0, used_steps - int(first_ready)) if first_ready is not None else 0

    return {
        "task_id": run["task_id"],
        "run_id": run["run_id"],
        "domain": run["domain"],
        "limit": limit,
        "steps": used_steps,
        "correct": correct,
        "premature_stop": premature_stop,
        "over_run": over_run,
        "wasted_steps": wasted,
    }


def sweep_fixed_limits(
    labelled_runs: Sequence[Dict[str, Any]],
    limits: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
) -> List[Dict[str, Any]]:
    """Evaluate all fixed step limits across a set of labelled runs."""
    results: List[Dict[str, Any]] = []
    for lim in limits:
        per_run = [evaluate_fixed_limit_on_run(r, lim) for r in labelled_runs]
        results.append({"limit": lim, "records": per_run})
    return results

