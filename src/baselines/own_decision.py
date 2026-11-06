"""Untrained agent's own stopping decision baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def evaluate_own_decision_on_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the agent's own natural stopping point on a labelled run."""
    used_steps = int(run.get("num_steps", len(run.get("steps", []))))
    correct = float(run.get("correct", 0))
    hindsight = run.get("hindsight", {})
    first_ready = hindsight.get("first_ready_step")

    premature_stop = int(
        hindsight.get("stopped_too_early", 0)
        or (first_ready is not None and used_steps < int(first_ready))
    )
    over_run = int(
        hindsight.get("continued_after_ready", 0)
        or (first_ready is not None and used_steps > int(first_ready))
    )
    wasted = int(hindsight.get("wasted_steps", 0))

    return {
        "task_id": run["task_id"],
        "run_id": run["run_id"],
        "domain": run["domain"],
        "steps": used_steps,
        "correct": correct,
        "premature_stop": premature_stop,
        "over_run": over_run,
        "wasted_steps": wasted,
    }


def evaluate_own_decision(labelled_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate own-decision baseline on all labelled runs."""
    return [evaluate_own_decision_on_run(r) for r in labelled_runs]

