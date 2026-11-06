"""Unit tests for answer checkers (Search QA, Maths, Code) and tools (Sandbox, Calculator)."""

from __future__ import annotations

import time

from src.tasks.code import check_code
from src.tasks.math import check_math
from src.tasks.search_qa import check_search_qa, compute_exact, compute_f1
from src.tools.calculator import calc
from src.tools.sandbox import run_code_sandbox


def test_search_qa_checker_handmade() -> None:
    # Exact match with article/punctuation/case differences
    res1 = check_search_qa("The United States of America!", "United States of America")
    assert res1["correct"] == 1
    assert res1["em"] == 1.0
    assert res1["f1"] == 1.0

    # Partial token overlap -> EM=0, 0 < F1 < 1
    res2 = check_search_qa("Scott Derrickson", "Scott Thomas Derrickson")
    assert res2["correct"] == 0
    assert res2["em"] == 0.0
    assert abs(res2["f1"] - 0.8) < 1e-5

    # Multiple gold aliases
    res3 = check_search_qa("NYC", ["New York City", "NYC", "New York"])
    assert res3["correct"] == 1
    assert compute_exact("yes", "Yes.") == 1.0
    assert compute_f1("Barack Obama", "President Barack Obama") == 0.8


def test_math_checker_handmade() -> None:
    assert check_math("The answer is $1,250.50.", "#### 1250.5")["correct"] == 1
    assert check_math("3/4", "0.75")["correct"] == 1
    assert check_math("-42", "-42.0000")["correct"] == 1
    assert check_math("41", "42")["correct"] == 0
    assert check_math("No numbers here", "10")["correct"] == 0


def test_code_checker_and_sandbox() -> None:
    good_code = """```python
def multiply(a, b):
    return a * b
```"""
    tests = ["assert multiply(3, 4) == 12", "assert multiply(-2, 5) == -10"]
    res_good = check_code(good_code, tests)
    assert res_good["correct"] == 1
    assert res_good["passed"] is True

    bad_code = "def multiply(a, b):\n    return a + b"
    res_bad = check_code(bad_code, tests)
    assert res_bad["correct"] == 0
    assert res_bad["passed"] is False


def test_sandbox_kills_infinite_loop_and_blocks_network() -> None:
    t0 = time.time()
    res_loop = run_code_sandbox("while True:\n    pass\n", timeout_seconds=1.5)
    elapsed = time.time() - t0
    assert res_loop["timed_out"] is True
    assert res_loop["ok"] is False
    assert elapsed < 4.0

    # Verify network is blocked inside sandbox
    net_code = "import socket\ns = socket.socket()\n"
    res_net = run_code_sandbox(net_code, timeout_seconds=2.0)
    assert res_net["ok"] is False
    assert "Network access is disabled" in res_net["stderr"]


def test_calculator_tool() -> None:
    c1 = calc("48 + 48 / 2")
    assert c1["ok"] is True
    assert c1["result"] == "72"

    c2 = calc("(2**10 - 24) / 100")
    assert c2["ok"] is True
    assert c2["result"] == "10"

    c3 = calc("1/3 + 1/6")
    assert c3["ok"] is True
    assert c3["result"] == "0.5"

