"""Run model-written Python in a restricted child process, to check code answers.

HONEST WARNING: this is a best-effort guard, NOT a real security sandbox. It (1) allows only a few
harmless imports and blocks open/eval/exec, (2) runs in a fresh process with an empty environment
(so it cannot see AWS credentials), in a temporary folder, (3) applies CPU/memory limits and a
timeout. For untrusted code in production use a container or microVM. Disable with --no-exec.
"""
import os
import re
import subprocess
import sys
import tempfile

import config

ALLOWED_IMPORTS = {"math", "re", "json", "collections", "itertools", "functools", "string", "typing",
                   "heapq", "bisect", "datetime", "statistics", "dataclasses", "operator", "decimal",
                   "fractions", "copy", "textwrap", "enum"}
BLOCKED_PATTERNS = [r"__import__", r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\(",
                    r"\binput\s*\(", r"\bglobals\s*\(", r"\bgetattr\s*\(", r"\bbreakpoint\s*\(",
                    r"__builtins__", r"__subclasses__"]

# Prepended to every script: hard limits applied by the OS inside the child process.
PRELUDE = """import resource
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
"""


def screen(code):
    """Return None if the code looks acceptable, else a short reason."""
    for match in re.finditer(r"^\s*(?:import|from)\s+([\w.]+)", code, flags=re.MULTILINE):
        module = match.group(1).split(".")[0]
        if module not in ALLOWED_IMPORTS:
            return f"import of '{module}' is not allowed"
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"blocked construct {pattern}"
    return None


def run_python(code, timeout=None, harness=""):
    """Return (ok, stdout, detail). ok=False on error, timeout or blocked code.

    Only `code` (written by the model) is screened. `harness` is OUR trusted test driver, appended after it.
    """
    timeout = timeout or config.CODE_TIMEOUT_SECONDS
    reason = screen(code)
    if reason:
        return False, "", f"not executed: {reason}"
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "snippet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(PRELUDE.format(cpu=int(timeout) + 1) + code + "\n" + harness)
        try:
            done = subprocess.run([sys.executable, "-I", script], cwd=tmp, env={"PYTHONHASHSEED": "0"},
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "", f"timed out after {timeout}s"
    if done.returncode != 0:
        last_line = (done.stderr.strip().splitlines() or ["error"])[-1]
        return False, done.stdout, f"error: {last_line[:160]}"
    return True, done.stdout, ""
