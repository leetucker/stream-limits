"""Portable launcher contracts plus explicit, real-kernel integration tests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from stream_limits.enforcement import UnsupportedEnforcement
from stream_limits.launcher import capabilities, run_limited
from stream_limits.testing import kernel_oom_proved


def python(code):
    return [sys.executable, "-c", code]


def test_imports_need_no_optional_or_numerical_packages():
    root = str(Path(__file__).resolve().parents[1])
    code = """
import builtins,os,sys
before=dict(os.environ)
original=builtins.__import__
blocked={'psutil','pytest','numpy','polars','numba','pyarrow','scipy'}
def checked(name,*args,**kwargs):
    if name.split('.')[0] in blocked: raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
import stream_limits,stream_limits.launcher,stream_limits.testing,stream_limits.enforcement
assert not blocked.intersection(sys.modules)
assert dict(os.environ)==before
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        env=dict(os.environ, PYTHONPATH=root),
        check=True,
    )


def test_monitor_preserves_environment_and_is_not_a_cap(memory_runner):
    before = dict(os.environ)
    result = memory_runner(
        python(
            "import os,json; x=bytearray(8*1024*1024); print(json.dumps(dict(os.environ)))"
        ),
        mode="monitor",
        memory_gb=0.001,
        env=dict(os.environ, STREAM_LIMITS_LAUNCH_TEST="yes"),
    )
    assert result.status == "ok" and result.returncode == 0
    assert json.loads(result.stdout)["STREAM_LIMITS_LAUNCH_TEST"] == "yes"
    assert dict(os.environ) == before
    assert result.kernel is None and result.peak_rss_bytes > 0


def test_watchdog_and_timeout_have_distinct_causes():
    result = run_limited(
        python("import time; x=bytearray(128*1024*1024); time.sleep(5)"),
        mode="watchdog",
        memory_gb=0.064,
        timeout=10,
    )
    assert result.status == "rss_limit_exceeded" and result.returncode != 0
    assert result.peak_rss_bytes > 64_000_000
    result = run_limited(
        python("import time; time.sleep(5)"), mode="monitor", timeout=0.2
    )
    assert result.status == "timeout" and result.returncode != 0


def test_output_does_not_deadlock_and_is_bounded():
    result = run_limited(
        python("import sys; print('x'*200000); print('y'*200000,file=sys.stderr)"),
        mode="monitor",
        max_output_bytes=128,
        timeout=10,
    )
    assert result.status == "ok"
    assert len(result.stdout.encode()) <= 128 and len(result.stderr.encode()) <= 128
    assert result.stdout_truncated and result.stderr_truncated


def test_command_failure_is_not_an_oom():
    result = run_limited(
        python("import sys; print('failure'); sys.exit(42)"), mode="monitor"
    )
    assert result.status == "command_failed" and result.returncode == 42
    assert "failure" in result.stdout


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "bad"},
        {"mode": "watchdog"},
        {"mode": "monitor", "memory_gb": 0},
        {"mode": "monitor", "memory_gb": True},
        {"mode": "monitor", "memory_gb": float("nan")},
        {"mode": "monitor", "timeout": 0},
        {"mode": "monitor", "sample_interval": 0},
        {"mode": "monitor", "max_output_bytes": -1},
        {"mode": "monitor", "cgroup_parent": "/unused"},
    ],
)
def test_invalid_options_do_not_start_a_command(options, tmp_path):
    marker = tmp_path / "launched"
    with pytest.raises(ValueError):
        run_limited(
            python(f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            **options,
        )
    assert not marker.exists()


def test_hard_mode_never_silently_falls_back(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    command = python(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    with pytest.raises(UnsupportedEnforcement):
        run_limited(command, memory_gb=0.5)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(UnsupportedEnforcement):
        run_limited(command, memory_gb=0.5, cgroup_parent=tmp_path)
    assert not marker.exists()
    assert capabilities()["hard_backends"] == []


def test_timeout_terminates_descendants():
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    result = run_limited(python(code), mode="monitor", timeout=3)
    assert result.status == "timeout"
    pid = int(result.stdout.strip())
    def gone():
        # the killed process can vanish between any two of these calls
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True

    for _ in range(100):
        if gone():
            break
        time.sleep(0.01)
    else:
        pytest.fail("descendant survived launcher timeout")


def test_cli_is_separate_and_preserves_command_exit_code():
    command = [
        sys.executable,
        "-m",
        "stream_limits.launcher",
        "--mode",
        "monitor",
        "--",
        *python("raise SystemExit(7)"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert (
        result.returncode == 7
        and json.loads(result.stdout)["status"] == "command_failed"
    )
    result = subprocess.run(
        [sys.executable, "-m", "stream_limits"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert isinstance(json.loads(result.stdout), dict)


@pytest.mark.parametrize(
    "code,kills,expected", [(0, 0, False), (-9, 0, False), (0, 1, False), (-9, 1, True)]
)
def test_accepted_property_or_signal_alone_is_not_proof(code, kills, expected):
    assert (
        kernel_oom_proved({"returncode": code}, {"events": {"oom_kill": kills}})
        is expected
    )


def test_real_kernel_cap_and_cleanup(verified_memory_parent, memory_runner):
    before = {p for p in verified_memory_parent.iterdir() if p.is_dir()}
    result = memory_runner(
        python("x=bytearray(16*1024*1024); print('ok')"),
        mode="hard",
        memory_gb=128 * 2**20 / 1e9,
        cgroup_parent=verified_memory_parent,
    )
    assert result.status == "ok" and result.kernel["events"].get("oom_kill", 0) == 0
    assert result.kernel["memory_max_bytes"] == 128 * 2**20
    assert result.kernel["effective_memory_max_bytes"] <= 128 * 2**20
    assert {p for p in verified_memory_parent.iterdir() if p.is_dir()} == before


def test_real_kernel_counts_descendant_oom(verified_memory_parent):
    code = "import subprocess,sys; p=subprocess.run([sys.executable,'-c','x=bytearray(256*1024*1024)']); print(p.returncode)"
    result = run_limited(
        python(code),
        mode="hard",
        memory_gb=64 * 2**20 / 1e9,
        cgroup_parent=verified_memory_parent,
        timeout=10,
    )
    assert result.status == "kernel_oom" and result.kernel["events"]["oom_kill"] > 0


def test_portable_resource_fixture_does_not_skip(resource_runner, resource_enforcement):
    result = resource_runner(
        python("x=bytearray(8*1024*1024); print('resource test')"), memory_gb=0.5
    )
    assert result.status == "ok"
    assert result.mode == resource_enforcement["mode"]
    assert "resource test" in result.stdout


def test_watchdog_cannot_claim_enforcement_when_sampling_is_denied(monkeypatch):
    def denied(self):
        raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil.Process, "memory_info", denied)
    result = run_limited(
        python("import time; time.sleep(10)"), mode="watchdog", memory_gb=1, timeout=3
    )
    assert result.status == "monitoring_failed"
    assert not result.rss_sampling_complete


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux cgroup filesystem validation"
)
def test_ordinary_files_cannot_masquerade_as_a_kernel_cap(tmp_path):
    from stream_limits.enforcement import MemoryCgroup

    (tmp_path / "memory.limit_in_bytes").write_text("67108864")
    with pytest.raises(UnsupportedEnforcement, match="genuine kernel"):
        MemoryCgroup(tmp_path, 64 * 2**20)
    assert [p.name for p in tmp_path.iterdir()] == ["memory.limit_in_bytes"]


def test_os_peak_is_exact_and_covers_short_commands():
    touch = "x=bytearray(200_000_000); x[::4096]=b'1'*len(x[::4096])"
    result = run_limited(python(touch), mode="monitor", timeout=30)
    assert result.status == "ok"
    assert result.os_peak_rss_bytes is not None
    assert result.os_peak_rss_bytes >= 200_000_000
    if os.name != "posix":   # Windows reads its peak while sampling
        return
    # too short for any sample: the sampled peak can be 0, the OS's is not
    quick = run_limited(python("pass"), mode="monitor", timeout=30)
    assert quick.os_peak_rss_bytes is not None and quick.os_peak_rss_bytes > 0


@pytest.mark.skipif(os.name != "posix", reason="wait4 covers waited-for descendants")
def test_os_peak_includes_waited_for_descendants():
    inner = "x=bytearray(300_000_000); x[::4096]=b'1'*len(x[::4096])"
    code = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{inner!r}])"
    result = run_limited(python(code), mode="monitor", timeout=30)
    assert result.status == "ok" and result.os_peak_rss_bytes >= 300_000_000


@pytest.mark.skipif(os.name != "posix", reason="POSIX reaping")
def test_child_is_reaped_once_and_its_pid_not_touched_after(monkeypatch):
    # Before the fix the tree was stopped twice, the second time after the
    # child was reaped, when its pid could belong to an unrelated process.
    reaped, looked_up = [], []
    real_wait4, real_waitpid, real_process = os.wait4, os.waitpid, psutil.Process

    def wait4(pid, options):
        out = real_wait4(pid, options)
        if out[0]:
            reaped.append(pid)
        return out

    def waitpid(pid, options):
        out = real_waitpid(pid, options)
        if out[0]:
            reaped.append(pid)
        return out

    class Process(real_process):
        def __init__(self, pid=None):
            if pid in reaped:
                looked_up.append(pid)
            super().__init__(pid)

    monkeypatch.setattr(os, "wait4", wait4)
    monkeypatch.setattr(os, "waitpid", waitpid)
    monkeypatch.setattr(psutil, "Process", Process)
    for mode, kw in (("monitor", {}), ("watchdog", {"memory_gb": 0.05})):
        code = "x=bytearray(100_000_000); import time; time.sleep(5)"
        result = run_limited(python(code), mode=mode, timeout=10, **kw)
        assert result.status in ("ok", "rss_limit_exceeded")
    assert not looked_up, "a reaped pid was looked up again"
    assert len(reaped) == 2


def test_threads_sets_every_thread_variable(memory_runner):
    from stream_limits import _THREAD_ENV

    result = memory_runner(
        python("import os,json; print(json.dumps(dict(os.environ)))"),
        mode="monitor",
        threads=3,
        env=dict(os.environ, KEEP_ME="1"),
    )
    child = json.loads(result.stdout)
    assert child["KEEP_ME"] == "1"
    for name in ("STREAM_LIMITS_MAX_THREADS", *_THREAD_ENV):
        assert child[name] == "3", name
    with pytest.raises(ValueError):
        run_limited(python("pass"), mode="monitor", threads=0)


def test_cli_without_capture_passes_output_through(tmp_path):
    report = tmp_path / "r.json"
    command = [
        sys.executable, "-m", "stream_limits.launcher", "--mode", "monitor",
        "--no-capture", "--report", str(report), "--threads", "2", "--",
        *python("import os; print('child says', os.environ['POLARS_MAX_THREADS'])"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == "child says 2"
    summary = json.loads(result.stderr)
    assert summary["status"] == "ok" and summary["stdout"] == ""
    assert json.loads(report.read_text()) == summary
