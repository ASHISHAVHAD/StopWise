"""Cut-off label builder: forces answers at intermediate steps and computes ready_fraction_t."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.agent.action_parser import parse_forced_answer
from src.agent.loop import LLMGenerator, check_task_answer
from src.agent.prompt import build_forced_answer_prompt
from src.labels.hindsight import compute_hindsight_metrics


def build_cutoff_prompts_for_run(
    run: Dict[str, Any],
    task: Dict[str, Any],
    max_cutoff_points: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build forced-answer prompts for each step t of a saved run (up to the step before ANSWER)."""
    steps = run.get("steps", [])
    if not steps:
        return []

    last_is_answer = steps[-1].get("action") == "ANSWER"
    cutoff_steps = steps[:-1] if last_is_answer else steps

    # If ablating number of cut-off points (E8: only 3 cut-off points per run instead of all)
    selected_indices = list(range(len(cutoff_steps)))
    if max_cutoff_points is not None and len(selected_indices) > max_cutoff_points:
        # Evenly spaced indices including first and last cutoff step
        if max_cutoff_points == 1:
            selected_indices = [len(cutoff_steps) - 1]
        else:
            selected_indices = sorted(
                list(
                    {
                        round(i * (len(cutoff_steps) - 1) / (max_cutoff_points - 1))
                        for i in range(max_cutoff_points)
                    }
                )
            )

    items: List[Dict[str, Any]] = []
    for idx in selected_indices:
        prefix_history = steps[: idx + 1]
        step_rec = steps[idx]
        prompt = build_forced_answer_prompt(task, prefix_history)
        items.append(
            {
                "run_id": run["run_id"],
                "task_id": run["task_id"],
                "domain": run["domain"],
                "step_index": step_rec["step_index"],
                "cutoff_pos": idx,
                "prompt": prompt,
                "state_text": step_rec["state_text"],
                "next_action_in_run": steps[idx + 1]["action"] if (idx + 1) < len(steps) else "ANSWER",
                "next_args_in_run": steps[idx + 1].get("args", "") if (idx + 1) < len(steps) else "",
                "next_think_in_run": steps[idx + 1].get("think", "") if (idx + 1) < len(steps) else "",
                "agent_readiness": step_rec.get("readiness", 0.5),
                "agent_confidence": step_rec.get("token_confidence", 0.5),
            }
        )
    return items


def label_runs_with_cutoffs(
    runs: Sequence[Dict[str, Any]],
    task_map: Dict[str, Dict[str, Any]],
    generator: LLMGenerator,
    forced_samples: int = 5,
    ready_threshold: float = 0.6,
    temperature: float = 0.7,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Run the cut-off test across a list of saved runs and return enriched run dicts with labels."""
    all_cutoff_requests: List[Dict[str, Any]] = []
    for run in runs:
        task = task_map[run["task_id"]]
        reqs = build_cutoff_prompts_for_run(run, task)
        all_cutoff_requests.extend(reqs)

    unique_prompts: List[str] = [req["prompt"] for req in all_cutoff_requests]

    gen_outs: List[Dict[str, Any]] = []
    if unique_prompts:
        domains_in_batch = {req["domain"] for req in all_cutoff_requests}
        max_toks = 96 if "code" in domains_in_batch else 28
        gen_outs = generator.generate_batch(
            unique_prompts,
            max_new_tokens=max_toks,
            temperature=temperature,
            return_confidence=False,
            batch_size=40,
            num_return_sequences=forced_samples,
        )

    # Group results back per cutoff request
    req_results: Dict[tuple, Dict[str, Any]] = {}
    cursor = 0
    for req in all_cutoff_requests:
        task = task_map[req["task_id"]]
        sample_outs = gen_outs[cursor : cursor + forced_samples]
        cursor += forced_samples

        sampled_answers: List[str] = []
        sample_correct: List[int] = []
        for s_out in sample_outs:
            ans = parse_forced_answer(s_out["text"])
            chk = check_task_answer(task, ans)
            sampled_answers.append(ans)
            sample_correct.append(int(chk["correct"]))

        ready_fraction = sum(sample_correct) / max(1, len(sample_correct))
        is_ready = ready_fraction >= ready_threshold
        key = (req["run_id"], req["step_index"])
        req_results[key] = {
            "step_index": req["step_index"],
            "sampled_answers": sampled_answers,
            "sample_correct": sample_correct,
            "ready_fraction": round(float(ready_fraction), 4),
            "label": "ready" if is_ready else "not_ready",
            "ready_int": int(is_ready),
        }

    # Attach step-level cut-off labels and hindsight metrics to each run
    labelled_runs: List[Dict[str, Any]] = []
    for run in runs:
        steps = run.get("steps", [])
        last_is_answer = bool(steps and steps[-1].get("action") == "ANSWER")
        cutoff_steps = steps[:-1] if last_is_answer else steps

        step_fractions: List[float] = []
        cutoff_records: List[Dict[str, Any]] = []
        for s in cutoff_steps:
            key = (run["run_id"], s["step_index"])
            c_info = req_results.get(
                key,
                {
                    "step_index": s["step_index"],
                    "sampled_answers": [],
                    "sample_correct": [],
                    "ready_fraction": 0.0,
                    "label": "not_ready",
                    "ready_int": 0,
                },
            )
            step_fractions.append(c_info["ready_fraction"])
            cutoff_records.append(c_info)

        hindsight = compute_hindsight_metrics(
            step_ready_fractions=step_fractions,
            final_correct=int(run["correct"]),
            total_steps=len(steps),
            ready_threshold=ready_threshold,
            forced_at_limit=bool(run.get("forced_at_limit", False)),
            last_step_is_answer=last_is_answer,
        )

        enriched = dict(run)
        enriched["cutoff_labels"] = cutoff_records
        enriched["hindsight"] = hindsight
        labelled_runs.append(enriched)

    return labelled_runs

