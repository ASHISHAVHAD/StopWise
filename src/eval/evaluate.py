"""High-level evaluation orchestrator for StopWise experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.baselines.own_decision import evaluate_own_decision_on_run
from src.eval.metrics import summarize_policy_records


def evaluate_online_runs_against_hindsight(
    online_runs: Sequence[Dict[str, Any]],
    reference_hindsight_map: Dict[str, int],
    fixed_limit_curve: Sequence[Tuple[float, float]] = (),
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a set of online agent runs (from Method A, B, or C) using task-level hindsight reference steps."""
    records: List[Dict[str, Any]] = []
    early_answer_step1_count = 0

    for r in online_runs:
        tid = r["task_id"]
        used_steps = int(r.get("num_steps", len(r.get("steps", []))))
        correct = float(r.get("correct", 0))
        ref_ready_step = reference_hindsight_map.get(tid)

        if used_steps == 1 and r.get("steps") and r["steps"][0].get("action") == "ANSWER":
            early_answer_step1_count += 1

        if ref_ready_step is not None:
            premature = int(used_steps < ref_ready_step and correct < 0.5)
            over_run = int(used_steps > ref_ready_step)
            wasted = max(0, used_steps - ref_ready_step)
        else:
            premature = int(not r.get("forced_at_limit", False) and correct < 0.5 and used_steps <= 2)
            over_run = 0
            wasted = 0

        rec = {
            "task_id": tid,
            "run_id": r["run_id"],
            "domain": r["domain"],
            "steps": used_steps,
            "correct": correct,
            "premature_stop": premature,
            "over_run": over_run,
            "wasted_steps": wasted,
            "early_answer_step1": int(used_steps == 1 and r.get("steps", [{}])[0].get("action") == "ANSWER"),
        }
        records.append(rec)

    summary = summarize_policy_records(records, fixed_limit_curve=fixed_limit_curve, seed=seed)
    summary["early_answer_rate"] = round(early_answer_step1_count / max(1, len(online_runs)), 4)
    return summary, records

