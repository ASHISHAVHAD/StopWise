"""Unit tests for the cut-off label builder, hindsight stopping oracle, and SFT dataset builder."""

from __future__ import annotations

from src.agent.action_parser import parse_agent_output
from src.labels.cutoff_labels import build_cutoff_prompts_for_run
from src.labels.hindsight import compute_hindsight_metrics
from src.train.sft import build_sft_examples_from_runs


def test_hindsight_metrics_synthetic_run() -> None:
    # Synthetic 5-step run where step 1 is not_ready (0.2), step 2 is ready (0.8),
    # step 3 is ready (0.6), step 4 is ready (1.0), and step 5 is ANSWER (correct=1).
    # Hindsight stop step should be step 2, and wasted steps = 5 - 2 = 3.
    m = compute_hindsight_metrics(
        step_ready_fractions=[0.2, 0.8, 0.6, 1.0],
        final_correct=1,
        total_steps=5,
        ready_threshold=0.6,
        forced_at_limit=False,
        last_step_is_answer=True,
    )
    assert m["ready_flags"] == [0, 1, 1, 1]
    assert m["hindsight_step"] == 2
    assert m["first_ready_step"] == 2
    assert m["wasted_steps"] == 3
    assert m["continued_after_ready"] == 1
    assert m["stopped_too_early"] == 0

    # Synthetic run where step 1 is ready (0.8), step 2 dips to not_ready (0.2),
    # step 3 is ready (0.8), step 4 is ANSWER (correct=1).
    # Earliest step that STAYS ready afterwards is step 3!
    m2 = compute_hindsight_metrics(
        step_ready_fractions=[0.8, 0.2, 0.8],
        final_correct=1,
        total_steps=4,
        ready_threshold=0.6,
        forced_at_limit=False,
        last_step_is_answer=True,
    )
    assert m2["hindsight_step"] == 3
    assert m2["first_ready_step"] == 1
    assert m2["wasted_steps"] == 1
    assert m2["continued_after_ready"] == 1

    # Synthetic run that stops too early at step 2 with wrong answer
    m3 = compute_hindsight_metrics(
        step_ready_fractions=[0.0],
        final_correct=0,
        total_steps=2,
        ready_threshold=0.6,
        forced_at_limit=False,
        last_step_is_answer=True,
    )
    assert m3["hindsight_step"] is None
    assert m3["stopped_too_early"] == 1


def test_cutoff_prompts_and_sft_builder_synthetic() -> None:
    task = {
        "task_id": "math_gsm8k_train_1",
        "domain": "math",
        "question": "What is 10 + 20?",
        "answer": "30",
    }
    synthetic_run = {
        "run_id": "run_1",
        "task_id": "math_gsm8k_train_1",
        "domain": "math",
        "correct": 1,
        "final_answer": "30",
        "num_steps": 3,
        "steps": [
            {
                "step_index": 1,
                "state_text": "prompt_step_1",
                "think": "Compute 10 + 20",
                "action": "CONTINUE",
                "args": "10 + 20",
                "tool_result": "30",
                "readiness": 0.4,
            },
            {
                "step_index": 2,
                "state_text": "prompt_step_2",
                "think": "Verify 30",
                "action": "VERIFY",
                "args": "30 - 10",
                "tool_result": "20",
                "readiness": 0.7,
            },
            {
                "step_index": 3,
                "state_text": "prompt_step_3",
                "think": "Done",
                "action": "ANSWER",
                "args": "",
                "tool_result": "",
                "answer": "30",
                "readiness": 0.95,
            },
        ],
        "cutoff_labels": [
            {"step_index": 1, "ready_fraction": 0.8, "label": "ready", "ready_int": 1},
            {"step_index": 2, "ready_fraction": 1.0, "label": "ready", "ready_int": 1},
        ],
    }

    reqs = build_cutoff_prompts_for_run(synthetic_run, task)
    assert len(reqs) == 2
    assert "Give your final answer now" in reqs[0]["prompt"]

    sft_ex = build_sft_examples_from_runs(
        [synthetic_run],
        {task["task_id"]: task},
        balance_classes=True,
        include_readiness=True,
    )
    assert len(sft_ex) > 0
    labels = [x["label"] for x in sft_ex]
    assert labels.count("ready") == labels.count("not_ready")


def test_action_parser() -> None:
    raw = "<think>Checking fact.</think>\n<action>GATHER</action>\n<args>Albert Einstein birth year</args>\n<ready>0.35</ready>"
    parsed = parse_agent_output(raw)
    assert parsed["valid_action"] is True
    assert parsed["action"] == "GATHER"
    assert parsed["args"] == "Albert Einstein birth year"
    assert abs(parsed["readiness"] - 0.35) < 1e-6

    raw_bad = "I think I will search for something without tags."
    parsed_bad = parse_agent_output(raw_bad)
    assert parsed_bad["valid_action"] is False
    assert parsed_bad["action"] == "INVALID"

