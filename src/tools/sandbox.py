"""Restricted Python execution sandbox with timeout, memory limits, temp dir, and no network."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


_SANDBOX_PREAMBLE = """# Restricted Sandbox Preamble
import resource as _resource
import socket as _socket

# 1. Enforce CPU and memory limits inside subprocess
try:
    _mem_bytes = {memory_limit_mb} * 1024 * 1024
    _resource.setrlimit(_resource.RLIMIT_AS, (_mem_bytes, _mem_bytes))
except Exception:
    pass

try:
    _resource.setrlimit(_resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
except Exception:
    pass

# 2. Block all network access
def _blocked_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled inside the StopWise sandbox.")

_socket.socket = _blocked_network
_socket.create_connection = _blocked_network
_socket.getaddrinfo = _blocked_network
"""


def run_code_sandbox(
    code: str,
    tests: Optional[List[str]] = None,
    setup_code: str = "",
    entry_point: str = "",
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 512,
) -> Dict[str, Any]:
    """Execute Python code (and optional unit tests) in an isolated temporary directory."""
    cpu_limit = max(1, int(timeout_seconds) + 1)
    preamble = _SANDBOX_PREAMBLE.format(memory_limit_mb=memory_limit_mb, cpu_limit=cpu_limit)

    script_parts = [preamble]
    if setup_code.strip():
        script_parts.append(setup_code.strip())
    script_parts.append(str(code).strip())

    has_tests = bool(tests and len(tests) > 0)
    if has_tests:
        script_parts.append("\n# --- Unit Tests ---")
        for t in tests or []:
            script_parts.append(str(t).strip())
        script_parts.append('print("__STOPWISE_ALL_TESTS_PASSED__")')

    full_script = "\n\n".join(script_parts) + "\n"

    with tempfile.TemporaryDirectory(prefix="stopwise_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(full_script, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "*",
            "http_proxy": "http://127.0.0.1:0",
            "https_proxy": "http://127.0.0.1:0",
        }

        proc = subprocess.Popen(
            [sys.executable, "-I", str(script_path)],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            stdout, stderr = proc.communicate()
            returncode = -9
            stderr = (stderr or "") + f"\nTimeoutError: Execution exceeded {timeout_seconds}s limit."

    tests_passed = False
    if has_tests and returncode == 0 and "__STOPWISE_ALL_TESTS_PASSED__" in stdout:
        tests_passed = True
        stdout = stdout.replace("__STOPWISE_ALL_TESTS_PASSED__", "").strip()
    elif not has_tests and returncode == 0:
        tests_passed = True

    return {
        "ok": returncode == 0 and not timed_out,
        "returncode": returncode,
        "timed_out": timed_out,
        "tests_passed": tests_passed,
        "stdout": stdout.strip()[:2000],
        "stderr": stderr.strip()[:2000],
    }


def run_code(code: str, tests: Optional[List[str]] = None, timeout_seconds: float = 5.0) -> str:
    """Agent-facing helper that formats sandbox output concisely."""
    res = run_code_sandbox(code=code, tests=tests, timeout_seconds=timeout_seconds)
    parts: List[str] = []
    if res["timed_out"]:
        parts.append(f"Status: TIMEOUT ({timeout_seconds}s)")
    elif res["returncode"] == 0:
        parts.append("Status: SUCCESS" + (" (All tests passed)" if tests else ""))
    else:
        parts.append(f"Status: ERROR (exit code {res['returncode']})")

    if res["stdout"]:
        parts.append(f"Stdout: {res['stdout'][:500]}")
    if res["stderr"]:
        parts.append(f"Stderr: {res['stderr'][:500]}")
    return " | ".join(parts)
