"""Sympy-based exact calculator tool."""

from __future__ import annotations

import re
from typing import Any, Dict

import sympy


def calc(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression using SymPy and return exact & decimal string."""
    expr_str = str(expression).strip()
    # Strip surrounding tags or quotes if present
    expr_str = re.sub(r"^['\"`]+|['\"`]+$", "", expr_str).strip()
    if "=" in expr_str and not any(op in expr_str for op in ["==", "!=", "<=", ">="]):
        # Take RHS or LHS if model wrote "x = 2 + 3"
        parts = expr_str.split("=")
        expr_str = parts[-1].strip() or parts[0].strip()
    expr_str = expr_str.replace("^", "**").replace(",", "")
    if not expr_str:
        return {"ok": False, "result": "", "error": "Empty expression"}

    try:
        sym_expr = sympy.sympify(expr_str, evaluate=True)
        exact_val = sympy.simplify(sym_expr)
        try:
            float_val = float(exact_val.evalf())
            if float_val.is_integer():
                formatted = str(int(float_val))
            else:
                formatted = f"{float_val:.6g}"
        except Exception:
            formatted = str(exact_val)
        return {
            "ok": True,
            "result": formatted,
            "exact": str(exact_val),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "result": "",
            "error": f"Calculator error: {type(e).__name__}: {e}",
        }
