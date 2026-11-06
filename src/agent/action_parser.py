"""Strict parser for StopWise agent actions, arguments, readiness scores, and answers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


VALID_ACTIONS = {"CONTINUE", "GATHER", "VERIFY", "ANSWER"}


def parse_agent_output(raw_text: str) -> Dict[str, Any]:
    """Parse model output into think, action, args, readiness score, answer, and validity flag."""
    text = str(raw_text).strip()

    # Stop at any hallucinated <result> tag if the model tried to generate tool output itself
    if "<result>" in text:
        text = text.split("<result>")[0].strip()

    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""

    action_match = re.search(r"<action>\s*([A-Za-z_]+)\s*</action>", text, flags=re.IGNORECASE)
    raw_action = action_match.group(1).upper().strip() if action_match else ""

    args_match = re.search(r"<args>(.*?)</args>", text, flags=re.DOTALL | re.IGNORECASE)
    args = args_match.group(1).strip() if args_match else ""

    ready_match = re.search(r"<ready>\s*([-+]?\d*\.?\d+)\s*</ready>", text, flags=re.IGNORECASE)
    readiness: Optional[float] = None
    if ready_match:
        try:
            val = float(ready_match.group(1))
            readiness = max(0.0, min(1.0, val))
        except ValueError:
            readiness = None

    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    if not answer and "<answer>" in text.lower():
        # Unclosed <answer> tag at end of generation
        idx = text.lower().rfind("<answer>")
        answer = text[idx + len("<answer>") :].strip()

    is_valid = raw_action in VALID_ACTIONS
    action = raw_action if is_valid else "INVALID"

    # If action is ANSWER but <answer> was placed inside <args>, recover it while noting if empty
    if action == "ANSWER" and not answer and args:
        answer = args

    # Default readiness if not parsed
    parsed_readiness = readiness if readiness is not None else (0.9 if action == "ANSWER" else 0.3)

    return {
        "think": think,
        "action": action,
        "raw_action": raw_action,
        "args": args,
        "readiness": float(parsed_readiness),
        "readiness_parsed": readiness is not None,
        "answer": answer,
        "valid_action": bool(is_valid),
        "cleaned_text": text,
    }


def parse_forced_answer(raw_text: str) -> str:
    """Extract the answer string from a forced-answer completion (which starts after `<answer>`)."""
    text = str(raw_text).strip()
    if "<answer>" in text.lower():
        m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    if "</answer>" in text.lower():
        idx = text.lower().find("</answer>")
        return text[:idx].strip()
    # Stop at next XML tag if any
    for tag in ["<think>", "<action>", "<result>", "Step "]:
        if tag in text:
            text = text.split(tag)[0].strip()
    return text.strip()
