"""Reusable enforcement verification with no estimator or pytest dependency."""

import json
import os
import sys

from .launcher import run_limited


def kernel_oom_proved(result, metrics):
    """A signal or accepted launch alone is not evidence of kernel enforcement."""
    code = result.returncode if hasattr(result, "returncode") else result["returncode"]
    return code != 0 and metrics["events"].get("oom_kill", 0) > 0


def verify_memory_limit(parent, *, python=None, env=None, timeout=15):
    """Verify a small successful allocation, inherited detection and kernel OOM.

    The deliberate OOM affects only a fresh 64 MiB child group. The delegate
    must allow at least 64 MiB. Raises if enforcement/detection is not proved.
    """
    python = python or sys.executable
    control = run_limited(
        [
            python,
            "-c",
            "import json,stream_limits; x=bytearray(16*1024*1024); print(json.dumps({'detected_memory_gb':stream_limits.detect_memory_gb(environ={})}))",
        ],
        mode="hard",
        cgroup_parent=parent,
        memory_gb=64 * 2**20 / 1e9,
        timeout=timeout,
        env=env,
    )
    try:
        detected = json.loads(control.stdout)["detected_memory_gb"]
    except (ValueError, KeyError, TypeError):
        detected = 0
    expected = control.kernel["effective_memory_max_bytes"]
    detection_ok = control.status == "ok" and abs(
        detected * 1e9 - expected
    ) < os.sysconf("SC_PAGE_SIZE")
    result = run_limited(
        [python, "-c", "x=bytearray(256*1024*1024)"],
        mode="hard",
        cgroup_parent=parent,
        memory_gb=64 * 2**20 / 1e9,
        timeout=timeout,
        env=env,
    )
    passed = detection_ok and kernel_oom_proved(result, result.kernel)
    report = {
        "passed": passed,
        "detection_ok": detection_ok,
        "detected_memory_gb": detected,
        "control": control.as_dict(),
        "result": result.as_dict(),
        "cgroup": result.kernel,
    }
    if not passed:
        raise AssertionError(
            "kernel memory-limit verification failed: " + json.dumps(report)
        )
    return report
