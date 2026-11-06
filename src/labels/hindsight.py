"""Hindsight stopping step and wasted-step computation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def compute_hindsight_metrics(
    step_ready_fractions: Sequence[float],
    final_correct: int,
    total_steps: int,
    ready_threshold: float = 0.6,
    forced_at_limit: bool = False,
    last_step_is_answer: bool = True,
) -> Dict[str, Any]:
    """Compute hindsight stop step, wasted steps, over-run flag, and early-stop flag for a run.

    Definitions (Section 4, Stage 3):
    - `ready` at step t iff `ready_fraction_t >= ready_threshold`.
    - `hindsight_step`: earliest step t (1-indexed) that is `ready` and stays `ready` for all
      subsequent steps (including the final step if the final step is ANSWER and correct=1, or
      across all cut-off steps if the final answer itself was right or all subsequent cut-offs
      are ready).
    - `wasted_steps`: `total_steps - hindsight_step` if `hindsight_step` exists, else 0.
    """
    ready_flags: List[bool] = [float(rf) >= ready_threshold for rf in step_ready_fractions]

    # Build full step-by-step readiness sequence of length `total_steps`
    full_ready: List[bool] = list(ready_flags)
    if last_step_is_answer and len(full_ready) < total_steps:
        # The final ANSWER step is ready iff final_correct == 1 (or if the preceding cut-off was ready and final is correct)
        full_ready.append(bool(final_correct == 1))

    n = len(full_ready)
    hindsight_step: Optional[int] = None
    for idx in range(n):
        if all(full_ready[j] for j in range(idx, n)):
            hindsight_step = idx + 1  # 1-indexed step
            break

    # Also compute earliest-ever ready step (even if a later step dipped)
    first_ready_step: Optional[int] = None
    for idx, is_r in enumerate(full_ready):
        if is_r:
            first_ready_step = idx + 1
            break

    if hindsight_step is not None:
        wasted_steps = max(0, total_steps - hindsight_step)
    elif first_ready_step is not None:
        # Run was ready at an intermediate step but continued and later lost readiness / answered wrong
        wasted_steps = max(0, total_steps - first_ready_step)
    else:
        wasted_steps = 0

    continued_after_ready = int(
        (first_ready_step is not None and first_ready_step < total_steps)
    )

    # Stopped too early: voluntarily answered (not forced at max_steps limit) when not ready and got it wrong
    stopped_too_early = int(
        (not forced_at_limit)
        and (final_correct == 0)
        and (first_ready_step is None or (len(ready_flags) > 0 and not ready_flags[-1]))
    )

    return {
        "ready_flags": [int(x) for x in ready_flags],
        "full_ready": [int(x) for x in full_ready],
        "hindsight_step": hindsight_step,
        "first_ready_step": first_ready_step,
        "wasted_steps": int(wasted_steps),
        "continued_after_ready": int(continued_after_ready),
        "stopped_too_early": int(stopped_too_early),
        "hindsight_correct": int(first_ready_step is not None or final_correct == 1),
    }
