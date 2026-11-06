"""Hindsight-best (oracle ceiling) stopping policy."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def evaluate_hindsight_best_on_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Oracle stopping rule: stop at `hindsight_step` (or `first_ready_step`) if the run was ever ready."""
    n_steps = int(run.get("num_steps", len(run.get("steps", []))))
    cutoffs = {int(c["step_index"]): c for c in run.get("cutoff_labels", [])}
    hindsight = run.get("hindsight", {})
    h_step = hindsight.get("hindsight_step") or hindsight.get("first_ready_step")

    if h_step is not None and int(h_step) <= n_steps:
        used_steps = int(h_step)
        if used_steps in cutoffs:
            correct = max(float(cutoffs[used_steps].get("ready_fraction", 1.0)), float(run.get("correct", 0)))
            # Since `h_step` is ready (>= 0.6 majority correct), majority vote forced answer is correct (1.0)
            correct = 1.0 if cutoffs[used_steps].get("ready_int", 0) == 1 else correct
        else:
            correct = float(run.get("correct", 1.0))
    else:
        used_steps = n_steps
        correct = float(run.get("correct", 0))

    return {
        "task_id": run["task_id"],
        "run_id": run["run_id"],
        "domain": run["domain"],
        "steps": used_steps,
        "correct": correct,
        "premature_stop": 0,
        "over_run": 0,
        "wasted_steps": 0,
    }


def evaluate_hindsight_best(labelled_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate oracle hindsight-best ceiling on all labelled runs."""
    return [evaluate_hindsight_best_on_run(r) for r in labelled_runs]

