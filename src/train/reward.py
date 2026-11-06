"""Exact cost-aware reward function for StopWise GRPO training (with ablation modes)."""

from __future__ import annotations

from typing import Any, Dict


def compute_effort(
    tool_calls: int,
    generated_tokens: int,
    max_steps: int = 10,
    max_tokens: int = 4000,
    token_weight: float = 0.001,
    effort_mode: str = "full",
) -> float:
    """Compute normalized effort clipped to [0, 1].

    Formula:
        effort = (tool_calls + 0.001 * generated_tokens) / (max_steps + 0.001 * max_tokens)
    """
    if effort_mode == "none":
        return 0.0
    if effort_mode == "steps_only":
        denom = float(max(1, max_steps))
        raw = float(tool_calls) / denom
        return float(min(1.0, max(0.0, raw)))

    denom = float(max_steps) + float(token_weight) * float(max_tokens)
    if denom <= 0.0:
        return 0.0
    raw = (float(tool_calls) + float(token_weight) * float(generated_tokens)) / denom
    return float(min(1.0, max(0.0, raw)))


def compute_reward(
    correct: int,
    tool_calls: int,
    generated_tokens: int,
    invalid_actions: int,
    lam: float = 0.1,
    max_steps: int = 10,
    max_tokens: int = 4000,
    invalid_penalty: float = 0.1,
    token_weight: float = 0.001,
    effort_mode: str = "full",
) -> Dict[str, float]:
    """Compute exact StopWise reward:

        reward = correct - lam * effort - 0.1 * invalid_actions

    where `correct in {0, 1}` and `effort` is clipped to [0, 1].
    """
    c_val = 1.0 if int(correct) == 1 else 0.0
    effort = compute_effort(
        tool_calls=tool_calls,
        generated_tokens=generated_tokens,
        max_steps=max_steps,
        max_tokens=max_tokens,
        token_weight=token_weight,
        effort_mode=effort_mode,
    )
    inv_pen = float(invalid_penalty) * float(max(0, int(invalid_actions)))
    reward = c_val - float(lam) * effort - inv_pen
    return {
        "reward": float(reward),
        "correct": float(c_val),
        "effort": float(effort),
        "invalid_penalty": float(inv_pen),
        "lambda": float(lam),
    }

