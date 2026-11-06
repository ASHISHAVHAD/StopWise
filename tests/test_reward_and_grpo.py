"""Unit tests for the exact StopWise reward formula, GRPO advantage normalization, and `<result>` masking."""

from __future__ import annotations

import math
import torch

from src.agent.readiness_head import ReadinessHead
from src.train.grpo import build_trajectory_token_mask, compute_grpo_group_advantages
from src.train.reward import compute_effort, compute_reward


def test_exact_reward_formula_hand_computed() -> None:
    # Formula:
    # effort = clip((tool_calls + 0.001 * generated_tokens) / (max_steps + 0.001 * max_tokens), 0, 1)
    # reward = correct - lam * effort - 0.1 * invalid_actions
    # With max_steps=10, max_tokens=4000 -> denominator = 10 + 4 = 14.0
    # Case 1: correct=1, tool_calls=3, generated_tokens=500, invalid_actions=0, lam=0.3
    # effort = (3 + 0.5) / 14.0 = 3.5 / 14.0 = 0.25
    # reward = 1.0 - 0.3 * 0.25 - 0.0 = 0.925
    r1 = compute_reward(
        correct=1,
        tool_calls=3,
        generated_tokens=500,
        invalid_actions=0,
        lam=0.3,
        max_steps=10,
        max_tokens=4000,
    )
    assert math.isclose(r1["effort"], 0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(r1["reward"], 0.925, rel_tol=1e-9, abs_tol=1e-9)

    # Case 2: correct=0, tool_calls=7, generated_tokens=0, invalid_actions=2, lam=0.6
    # effort = 7.0 / 14.0 = 0.5
    # reward = 0.0 - 0.6 * 0.5 - 0.1 * 2 = -0.3 - 0.2 = -0.5
    r2 = compute_reward(
        correct=0,
        tool_calls=7,
        generated_tokens=0,
        invalid_actions=2,
        lam=0.6,
        max_steps=10,
        max_tokens=4000,
    )
    assert math.isclose(r2["effort"], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(r2["reward"], -0.5, rel_tol=1e-9, abs_tol=1e-9)

    # Case 3: clipping effort above 1.0
    e_clipped = compute_effort(tool_calls=20, generated_tokens=9000, max_steps=10, max_tokens=4000)
    assert e_clipped == 1.0


def test_grpo_advantages_and_result_masking() -> None:
    adv = compute_grpo_group_advantages([1.0, 1.0, 0.0, 0.0])
    assert len(adv) == 4
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-5)
    assert adv[0] > 0 and adv[2] < 0

    # Dummy word-level tokenizer to verify `<result>...</result>` span masking
    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [ord(c) for c in text]

    tok = DummyTokenizer()
    prompt = "Task: test\nStep 1:\n"
    agent_step1 = "<think>t</think><action>GATHER</action><args>q</args><ready>0.20</ready>\n"
    tool_res = "<result>Retrieved secret tool passage text</result>\n\nStep 2:\n"
    agent_step2 = "<think>done</think><action>ANSWER</action><args></args><ready>0.95</ready><answer>42</answer>"
    full = prompt + agent_step1 + tool_res + agent_step2

    masked = build_trajectory_token_mask(full, prompt, tok)
    in_ids = masked["input_ids"].tolist()
    mask = masked["action_mask"].tolist()
    assert len(in_ids) == len(mask)

    # Reconstruct text where mask == 1 and verify no `<result>` content is unmasked
    unmasked_chars = "".join(chr(i) for i, m in zip(in_ids, mask) if m == 1.0)
    assert "Retrieved secret tool passage text" not in unmasked_chars
    assert "<action>GATHER</action>" in unmasked_chars
    assert "<action>ANSWER</action>" in unmasked_chars


def test_readiness_head() -> None:
    head = ReadinessHead(hidden_size=16)
    dummy_hidden = torch.randn(4, 10, 16)
    preds = head(dummy_hidden)
    assert preds.shape == (4,)
    assert torch.all((preds >= 0.0) & (preds <= 1.0))
    targets = torch.tensor([0.0, 0.4, 0.8, 1.0])
    loss = head.compute_loss(preds, targets)
    assert loss.item() > 0.0
