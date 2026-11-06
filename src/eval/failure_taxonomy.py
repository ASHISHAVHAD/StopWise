"""Failure taxonomy classifier (E10) for test-set errors across all 6 specified categories."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


FAILURE_CATEGORIES = [
    "repeated_search",
    "reopened_source",
    "repeated_check",
    "repeated_calculation",
    "stopped_too_early",
    "looped_until_limit",
]


def _normalize_arg(s: str) -> str:
    return " ".join(re.findall(r"\w+", str(s).lower()))


def _jaccard_tokens(a: str, b: str) -> float:
    ta = set(_normalize_arg(a).split())
    tb = set(_normalize_arg(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def classify_failed_run(run: Dict[str, Any], max_steps: int = 10) -> str:
    """Rule-based classifier mapping a failed trajectory (`correct == 0`) to one of the 6 categories:
    - `repeated_search`
    - `reopened_source`
    - `repeated_check`
    - `repeated_calculation`
    - `stopped_too_early`
    - `looped_until_limit`
    """
    steps = run.get("steps", [])
    domain = run.get("domain", "search_qa")
    n_steps = len(steps)
    forced = bool(run.get("forced_at_limit", False))

    # Collect tool actions & arguments
    tool_steps = [s for s in steps if s.get("action") != "ANSWER"]

    # 1. Check reopened source (explicit `read:` of same doc or retrieving exact same top doc ID repeatedly)
    seen_doc_ids: set[str] = set()
    reopened = False
    for s in tool_steps:
        arg = str(s.get("args", "")).strip()
        res = str(s.get("tool_result", ""))
        if arg.lower().startswith("read"):
            doc_id = _normalize_arg(arg.replace("read", ""))
            if doc_id in seen_doc_ids:
                reopened = True
                break
            seen_doc_ids.add(doc_id)
        # Also check if top retrieved passage ID in `[doc_id]` was already read and then explicitly re-queried
        doc_matches = re.findall(r"\[([a-zA-Z0-9_]+)\]", res)
        if doc_matches and s.get("action") == "VERIFY":
            if doc_matches[0] in seen_doc_ids:
                reopened = True
                break
        for dm in doc_matches[:1]:
            seen_doc_ids.add(dm)

    # 2. Check repeated check (`VERIFY` used multiple times or repeating previous check)
    verify_steps = [s for s in tool_steps if s.get("action") == "VERIFY"]
    if len(verify_steps) >= 2:
        return "repeated_check"
    if len(verify_steps) == 1 and len(tool_steps) >= 2:
        v_arg = str(verify_steps[0].get("args", ""))
        for other in tool_steps:
            if other is not verify_steps[0] and _jaccard_tokens(v_arg, str(other.get("args", ""))) >= 0.65:
                return "repeated_check"

    if reopened:
        return "reopened_source"

    # 3. Check repeated search or repeated calculation across tool steps
    prev_args: List[str] = []
    prev_results: List[str] = []
    has_repeat = False
    for s in tool_steps:
        arg = str(s.get("args", ""))
        res = str(s.get("tool_result", ""))
        for pa, pr in zip(prev_args, prev_results):
            if _jaccard_tokens(arg, pa) >= 0.6 or (res and pr and res == pr):
                has_repeat = True
                break
        prev_args.append(arg)
        prev_results.append(res)

    if has_repeat:
        if domain == "math":
            return "repeated_calculation"
        elif domain == "search_qa":
            return "repeated_search"
        else:
            return "repeated_check"

    # 4. Check looped until the limit
    if forced or n_steps >= max_steps:
        return "looped_until_limit"

    # 5. Check stopped too early (answered voluntarily at step 1 or 2, or before hindsight ready)
    hindsight = run.get("hindsight", {})
    if n_steps <= 2 or hindsight.get("stopped_too_early", 0) == 1:
        return "stopped_too_early"

    # 6. Multi-step failed trajectory that didn't hit max_steps: classify by domain-dominant redundancy
    if domain == "search_qa":
        return "repeated_search"
    elif domain == "math":
        return "repeated_calculation"
    else:
        return "repeated_check"


def analyze_failures(
    runs: Sequence[Dict[str, Any]],
    method_name: str,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """Classify all failed test runs (`correct == 0`) for a method and return counts & shares."""
    failed = [r for r in runs if int(r.get("correct", 0)) == 0]
    counts = {cat: 0 for cat in FAILURE_CATEGORIES}
    classified_examples: List[Dict[str, Any]] = []

    for r in failed:
        cat = classify_failed_run(r, max_steps=max_steps)
        counts[cat] = counts.get(cat, 0) + 1
        classified_examples.append(
            {
                "run_id": r.get("run_id", ""),
                "task_id": r.get("task_id", ""),
                "domain": r.get("domain", ""),
                "num_steps": r.get("num_steps", len(r.get("steps", []))),
                "forced_at_limit": bool(r.get("forced_at_limit", False)),
                "category": cat,
                "question": r.get("question", "")[:120],
                "final_answer": str(r.get("final_answer", ""))[:80],
                "gold_answer": str(r.get("gold_answer", ""))[:80],
            }
        )

    n_total = max(1, len(runs))
    n_fail = len(failed)
    row: Dict[str, Any] = {
        "method": method_name,
        "total_runs": len(runs),
        "total_failures": n_fail,
        "failure_rate": round(n_fail / n_total, 4),
    }
    for cat in FAILURE_CATEGORIES:
        row[f"{cat}_count"] = counts[cat]
        row[f"{cat}_share_of_failures"] = round(counts[cat] / max(1, n_fail), 4)
        row[f"{cat}_rate_overall"] = round(counts[cat] / n_total, 4)

    return {"summary": row, "examples": classified_examples}
