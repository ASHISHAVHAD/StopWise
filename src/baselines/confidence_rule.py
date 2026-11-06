"""Confidence / readiness threshold stopping rule baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def evaluate_confidence_rule_on_run(
    run: Dict[str, Any],
    threshold: float,
    score_key: str = "readiness",
) -> Dict[str, Any]:
    """Stop at the first step t where `step[score_key] >= threshold` (or natural stop if earlier).

    `score_key` can be `'readiness'` (parsed `<ready>0.xx</ready>`) or `'token_confidence'` (mean token probability).
    """
    steps = run.get("steps", [])
    n_steps = len(steps)
    cutoffs = {int(c["step_index"]): c for c in run.get("cutoff_labels", [])}
    hindsight = run.get("hindsight", {})
    first_ready = hindsight.get("first_ready_step")

    chosen_step = n_steps
    stopped_by_rule = False
    for s in steps:
        s_idx = int(s["step_index"])
        if s.get("action") == "ANSWER":
            chosen_step = s_idx
            break
        score = float(s.get(score_key, s.get("readiness", 0.5)))
        if score >= threshold:
            chosen_step = s_idx
            stopped_by_rule = True
            break

    if not stopped_by_rule or chosen_step >= n_steps:
        used_steps = n_steps
        correct = float(run.get("correct", 0))
    else:
        used_steps = chosen_step
        if chosen_step in cutoffs:
            correct = float(cutoffs[chosen_step].get("ready_fraction", 0.0))
        else:
            correct = float(run.get("correct", 0))

    premature_stop = int(
        (first_ready is not None and used_steps < int(first_ready))
        or (first_ready is None and correct < 0.5 and used_steps < n_steps)
    )
    over_run = int(first_ready is not None and used_steps > int(first_ready))
    wasted = max(0, used_steps - int(first_ready)) if first_ready is not None else 0

    return {
        "task_id": run["task_id"],
        "run_id": run["run_id"],
        "domain": run["domain"],
        "threshold": threshold,
        "score_key": score_key,
        "steps": used_steps,
        "correct": correct,
        "premature_stop": premature_stop,
        "over_run": over_run,
        "wasted_steps": wasted,
    }


def sweep_confidence_thresholds(
    labelled_runs: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    score_key: str = "readiness",
) -> List[Dict[str, Any]]:
    """Sweep stopping thresholds across all labelled runs."""
    results: List[Dict[str, Any]] = []
    for tau in thresholds:
        per_run = [evaluate_confidence_rule_on_run(r, tau, score_key=score_key) for r in labelled_runs]
        results.append({"threshold": tau, "score_key": score_key, "records": per_run})
    return results

