"""Maths task loader (GSM8K) and numeric answer checker."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import sympy


def extract_last_number(text: str) -> Optional[str]:
    """Extract the final number or fraction from a text string."""
    if not text:
        return None
    s = str(text).strip()
    # Look for #### pattern first (GSM8K style)
    if "####" in s:
        s = s.split("####")[-1].strip()
    # Remove thousands commas inside numbers (e.g., 1,234 -> 1234)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # Match fractions like -3/4 or decimals/integers like -12.34
    matches = re.findall(r"[-+]?\d+\s*/\s*\d+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not matches:
        return None
    return matches[-1].replace(" ", "")


def parse_numeric_value(val_str: Optional[str]) -> Optional[float]:
    """Parse a numeric string (including fractions) into float."""
    if val_str is None:
        return None
    try:
        return float(val_str)
    except ValueError:
        try:
            expr = sympy.sympify(val_str)
            return float(expr.evalf())
        except Exception:
            return None


def check_math(prediction: str, gold_answer: str, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> Dict[str, Any]:
    """Check maths prediction by comparing normalized final numeric values."""
    pred_str = extract_last_number(prediction)
    gold_str = extract_last_number(gold_answer)

    pred_val = parse_numeric_value(pred_str)
    gold_val = parse_numeric_value(gold_str)

    correct = False
    if pred_val is not None and gold_val is not None:
        if math.isfinite(pred_val) and math.isfinite(gold_val):
            correct = math.isclose(pred_val, gold_val, rel_tol=rel_tol, abs_tol=abs_tol)
    elif pred_str is not None and gold_str is not None:
        correct = pred_str.strip() == gold_str.strip()

    return {
        "correct": int(correct),
        "pred_number": pred_val,
        "gold_number": gold_val,
        "pred_str": pred_str,
        "gold_str": gold_str,
    }


def extract_gsm8k_item(row: Dict[str, Any], idx: int, split_name: str) -> Dict[str, Any]:
    """Convert a GSM8K row into standardized task dict."""
    question = str(row["question"]).strip()
    full_solution = str(row["answer"]).strip()
    gold_num_str = extract_last_number(full_solution) or full_solution
    return {
        "task_id": f"math_gsm8k_{split_name}_{idx}",
        "domain": "math",
        "source": "gsm8k",
        "split": split_name,
        "question": question,
        "answer": gold_num_str,
        "full_solution": full_solution,
    }

