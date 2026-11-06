"""Code task loader (MBPP, HumanEval) and sandbox-based unit test checker."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.tools.sandbox import run_code_sandbox


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code fences if present, else raw text."""
    s = str(text).strip()
    # Look for ```python ... ``` or ``` ... ```
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    # Strip trailing tags if any
    s = re.sub(r"</?(?:answer|code)>", "", s, flags=re.IGNORECASE).strip()
    return s


def check_code(
    prediction: str,
    test_list: List[str],
    test_setup_code: str = "",
    entry_point: str = "",
    timeout_seconds: int = 5,
    memory_limit_mb: int = 512,
) -> Dict[str, Any]:
    """Run candidate Python code against all unit tests inside the restricted sandbox."""
    candidate_code = extract_python_code(prediction)
    if not candidate_code:
        return {
            "correct": 0,
            "passed": False,
            "stdout": "",
            "stderr": "Empty candidate code",
            "returncode": -1,
        }

    res = run_code_sandbox(
        code=candidate_code,
        tests=test_list,
        setup_code=test_setup_code,
        entry_point=entry_point,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )
    passed = bool(res["returncode"] == 0 and res.get("tests_passed", False))
    return {
        "correct": int(passed),
        "passed": passed,
        "stdout": res["stdout"],
        "stderr": res["stderr"],
        "returncode": res["returncode"],
    }


def extract_mbpp_item(row: Dict[str, Any], idx: int, split_name: str) -> Dict[str, Any]:
    """Convert an MBPP row into standardized task dict."""
    task_id = row.get("task_id", idx)
    prompt = str(row.get("prompt", row.get("text", ""))).strip()
    code = str(row.get("code", "")).strip()
    test_list = [str(t) for t in row.get("test_list", [])]
    setup_code = "\n".join(str(x) for x in row.get("test_imports", [])) if "test_imports" in row else str(row.get("test_setup_code", ""))

    # Extract function signature from reference code to show the expected function name/signature in prompt
    sig_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*:", code)
    signature_hint = ""
    func_name = ""
    if sig_match:
        func_name = sig_match.group(1)
        signature_hint = f"Write a Python function `{func_name}({sig_match.group(2)})`."
    sample_test = test_list[0] if test_list else ""

    full_question = f"{prompt}\n{signature_hint}".strip()
    if sample_test:
        full_question += f"\nExample test: `{sample_test}`"

    return {
        "task_id": f"code_mbpp_{split_name}_{task_id}",
        "domain": "code",
        "source": "mbpp",
        "split": split_name,
        "question": full_question,
        "answer": code,
        "test_list": test_list,
        "test_setup_code": setup_code,
        "entry_point": func_name,
    }


def extract_humaneval_item(row: Dict[str, Any], idx: int, split_name: str = "test") -> Dict[str, Any]:
    """Convert a HumanEval row into standardized task dict."""
    task_id = str(row.get("task_id", f"HumanEval/{idx}")).replace("/", "_")
    prompt = str(row["prompt"]).strip()
    canonical_solution = str(row["canonical_solution"])
    test_code = str(row["test"]).strip()
    entry_point = str(row["entry_point"]).strip()

    full_test = f"{test_code}\ncheck({entry_point})\n"
    question = (
        f"Complete the following Python function `{entry_point}` (return the full function implementation including imports if needed):\n\n"
        f"```python\n{prompt}\n```"
    )
    return {
        "task_id": f"code_humaneval_{split_name}_{task_id}",
        "domain": "code",
        "source": "humaneval",
        "split": split_name,
        "question": question,
        "answer": f"{prompt}\n{canonical_solution}",
        "test_list": [full_test],
        "test_setup_code": "",
        "entry_point": entry_point,
    }

