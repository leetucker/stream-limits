"""Opt-in subprocess monitoring, sampled watchdogs and Linux kernel memory caps.

Install stream-limits[launcher] for psutil. Neither importing this module nor
running a command imports a numerical library. Existing stream-limits defaults
and the parent process's environment are unchanged.
"""

import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from . import _THREAD_ENV
from .enforcement import MemoryCgroup, UnsupportedEnforcement


@dataclass
class RunResult:
    command: list[str]
    mode: str
    status: str
    returncode: int
    elapsed_s: float
    requested_memory_bytes: int | None
    peak_rss_bytes: int
    kernel: dict | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    descendants_terminated: bool
    rss_sampling_complete: bool
    os_peak_rss_bytes: int | None = None
    peak_private_bytes: int | None = None
    memory_measure: str = "rss"

    # os_peak_rss_bytes: the largest resident set any single process reached,
    # as the OS records it. POSIX: the child's ru_maxrss from wait4, which
    # covers the child and the descendants it waited for (exact, not sampled;
    # not summed over the tree; background processes it left running are not
    # included). Windows: the largest peak working set read while sampling.
    # None when unavailable. Like RSS, it counts pages of memory-mapped files.
    #
    # peak_private_bytes: the largest sampled total over the process tree of
    # memory that is not a file's pages (see _private_bytes). None on macOS.
    #
    # memory_measure: what the watchdog compared with the limit, "private" or
    # "rss" (asked for, or private memory is unavailable on this platform).

    def as_dict(self):
        return asdict(self)


def capabilities():
    """Backend availability, not a claim that a controller is delegated."""
    return {
        "platform": sys.platform,
        "psutil_available": importlib.util.find_spec("psutil") is not None,
        "monitor": True,
        "watchdog": True,
        "private_memory": private_memory_available(),
        "hard_backends": ["cgroup-v1", "cgroup-v2"] if sys.platform == "linux" else [],
        "hard_limit_accounting": "cgroup-charged memory"
        if sys.platform == "linux"
        else None,
        "note": "Monitoring/watchdogs require the launcher extra; hard mode requires a delegated Linux memory controller.",
    }


def thread_env(threads, env=None):
    """A copy of `env` (default os.environ) with STREAM_LIMITS_MAX_THREADS and
    every thread variable stream-limits knows (Polars, numba, OpenMP, BLAS,
    numexpr, rayon, Julia) set to `threads`."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    out = dict(os.environ if env is None else env)
    for name in ("STREAM_LIMITS_MAX_THREADS", *_THREAD_ENV):
        out[name] = str(threads)
    return out


def _private_bytes(process, info):
    """Memory of one process that is not a file's pages, which the kernel can
    drop and read back (memory-mapped scratch files, mapped libraries).
    Linux: anonymous plus shared memory (RssAnon + RssShmem; shared memory
    is not reclaimable without swap), else RSS less psutil's shared pages.
    Windows: private bytes (committed memory, so it can exceed RSS). None
    elsewhere (macOS has no cheap equivalent)."""
    if sys.platform.startswith("linux"):
        try:
            found = {}
            with open(f"/proc/{process.pid}/status", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith(("RssAnon:", "RssShmem:")):
                        key, value = line.split()[:2]
                        found[key] = int(value) * 1024
            if "RssAnon:" in found:
                return found["RssAnon:"] + found.get("RssShmem:", 0)
        except (OSError, ValueError):
            pass
        return max(0, info.rss - getattr(info, "shared", 0))
    if hasattr(info, "private"):
        return info.private
    return None


def private_memory_available():
    """Whether this platform reports private memory (Linux, Windows)."""
    return sys.platform.startswith("linux") or os.name == "nt"


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def run_limited(
    command: Sequence[str],
    *,
    memory_gb: float | None = None,
    mode="hard",
    cgroup_parent=None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd=None,
    sample_interval=0.01,
    max_output_bytes=1_000_000,
    threads: int | None = None,
    capture_output: bool = True,
    memory_measure: str = "private",
):
    """Run a command without a shell and return measurements and termination cause.

    hard: verified Linux cgroup cap, including descendants; no automatic fallback.
    watchdog: sampled process-tree memory threshold, which can overshoot.
    monitor: measurements/timeouts only, without a memory ceiling.

    memory_measure: what the watchdog compares with the limit. "private"
    (default) leaves out pages of files, which the kernel can reclaim, so a
    program that memory-maps large scratch files is not stopped for them;
    where private memory is unavailable (macOS) RSS is used, and the result's
    memory_measure says so. "rss" counts every resident page.

    GB is decimal. The supervisor stays outside the child's hard cap. env has
    subprocess replacement semantics; no budget/thread variables are injected
    unless `threads` is given, which sets them all (`thread_env`). Captured
    output uses temporary disk files; only a bounded tail is retained. With
    capture_output=False the child writes to this process's stdout/stderr
    and the result's stdout/stderr are empty.
    """
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(v, str) or not v for v in command)
    ):
        raise ValueError(
            "command must be a nonempty sequence of nonempty argument strings"
        )
    if mode not in ("hard", "watchdog", "monitor"):
        raise ValueError("mode must be hard, watchdog or monitor")
    if memory_measure not in ("private", "rss"):
        raise ValueError("memory_measure must be private or rss")
    measure = (
        "private"
        if memory_measure == "private" and private_memory_available()
        else "rss"
    )
    if memory_gb is not None:
        _positive(memory_gb, "memory_gb")
    if mode != "monitor" and memory_gb is None:
        raise ValueError("hard/watchdog mode requires memory_gb")
    if timeout is not None:
        _positive(timeout, "timeout")
    _positive(sample_interval, "sample_interval")
    if threads is not None:
        env = thread_env(threads, env)
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 0
    ):
        raise ValueError("max_output_bytes must be a nonnegative integer")
    if mode == "hard" and (sys.platform != "linux" or cgroup_parent is None):
        raise UnsupportedEnforcement(
            "hard mode requires Linux and an explicit delegated cgroup_parent; no fallback was used"
        )
    if mode != "hard" and cgroup_parent is not None:
        raise ValueError("cgroup_parent is only used in hard mode")
    try:
        import psutil
    except ImportError as exc:
        raise ImportError(
            "install stream-limits[launcher] to use the launcher"
        ) from exc
    limit = int(memory_gb * 1e9) if memory_gb is not None else None
    if limit is not None and limit < 1:
        raise ValueError("memory_gb must represent at least one byte")
    group = MemoryCgroup(cgroup_parent, limit) if mode == "hard" else None
    proc, kernel, tracked = None, None, {}
    reason, peak, descendants = None, 0, False
    private_peak = 0 if private_memory_available() else None
    os_peak, stopped = None, False
    sampling_complete = True
    started = time.monotonic()
    posix = os.name == "posix"

    def exited():
        # POSIX: WNOWAIT leaves the child a zombie, so its pid (and process
        # group id) cannot be reused before stop_tree kills the group and
        # reaps it.
        if posix and hasattr(os, "waitid"):
            return (
                os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is not None
            )
        if posix:
            # macOS builds of Python often lack os.waitid. Only this process
            # can reap its child, so a zombie status also means "exited, pid
            # still held".
            if process is None:
                return True
            try:
                return process.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True
        return proc.poll() is not None

    def stop_tree():
        # Once only: after the child is reaped its pid may belong to another
        # process.
        nonlocal descendants, os_peak, stopped
        if proc is None or stopped:
            return
        stopped = True
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
            tracked.update({(p.pid, p.create_time()): p for p in children})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for p in tracked.values():
            try:
                if p.is_running():
                    p.kill()
                    descendants = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if posix:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            # ru_maxrss: kilobytes on Linux, bytes on macOS
            os_peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        else:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def tail(handle):
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - max_output_bytes))
        return handle.read().decode("utf-8", "replace"), size > max_output_bytes

    try:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            actual = group.command(command) if group else list(command)
            options = (
                {"start_new_session": True}
                if os.name == "posix"
                else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            )
            streams = (
                {"stdout": out, "stderr": err} if capture_output else {}
            )
            proc = subprocess.Popen(actual, env=env, cwd=cwd, **streams, **options)
            try:
                process = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                process = None
            while not exited():
                rss, private = 0, 0
                try:
                    members = (
                        [process, *process.children(recursive=True)]
                        if process is not None
                        else []
                    )
                    for p in members:
                        try:
                            info = p.memory_info()
                            rss += info.rss
                            if private_peak is not None:
                                private += _private_bytes(p, info) or 0
                            if hasattr(info, "peak_wset"):  # Windows
                                os_peak = max(os_peak or 0, info.peak_wset)
                            if p.pid != proc.pid:
                                tracked[(p.pid, p.create_time())] = p
                        except psutil.NoSuchProcess:
                            pass
                        except psutil.AccessDenied:
                            sampling_complete = False
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    sampling_complete = False
                peak = max(peak, rss)
                if private_peak is not None:
                    private_peak = max(private_peak, private)
                if mode == "watchdog" and not sampling_complete:
                    reason = "monitoring_failed"
                    break
                watched = private if measure == "private" else rss
                if mode == "watchdog" and watched > limit:
                    reason = "rss_limit_exceeded"
                    break
                if timeout is not None and time.monotonic() - started >= timeout:
                    reason = "timeout"
                    break
                remaining = (
                    timeout - (time.monotonic() - started)
                    if timeout is not None
                    else sample_interval
                )
                time.sleep(min(sample_interval, max(0, remaining)))
            stop_tree()
            if group:
                # Also clean up descendants that outlive a short-lived parent
                # and therefore escape process-tree sampling.
                descendants |= bool((group.path / "cgroup.procs").read_text().split())
                if descendants:
                    for pid in (group.path / "cgroup.procs").read_text().split():
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                kernel = group.metrics()
            if capture_output:
                stdout, out_truncated = tail(out)
                stderr, err_truncated = tail(err)
            else:
                stdout, out_truncated, stderr, err_truncated = "", False, "", False
            status = (
                "kernel_oom"
                if kernel and kernel["events"].get("oom_kill", 0)
                else reason or ("ok" if proc.returncode == 0 else "command_failed")
            )
            return RunResult(
                list(command),
                mode,
                status,
                proc.returncode,
                time.monotonic() - started,
                limit,
                peak,
                kernel,
                stdout,
                stderr,
                out_truncated,
                err_truncated,
                descendants,
                sampling_complete,
                os_peak_rss_bytes=os_peak,
                peak_private_bytes=private_peak,
                memory_measure=measure,
            )
    finally:
        try:
            if proc is not None:
                stop_tree()
        finally:
            if group:
                group.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["hard", "watchdog", "monitor"], default="hard"
    )
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--cgroup-parent")
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--threads", type=int, help="set every thread variable in the child to this"
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="let the command write to this terminal; the JSON summary goes to stderr",
    )
    parser.add_argument("--report", help="also write the JSON summary to this file")
    parser.add_argument(
        "--memory-measure",
        choices=["private", "rss"],
        default="private",
        help="what the watchdog limits: private memory (default; not file pages) or RSS",
    )
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.capabilities:
        print(json.dumps(capabilities(), indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = run_limited(
            command,
            memory_gb=args.memory_gb,
            mode=args.mode,
            cgroup_parent=args.cgroup_parent,
            timeout=args.timeout,
            threads=args.threads,
            capture_output=not args.no_capture,
            memory_measure=args.memory_measure,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    summary = json.dumps(result.as_dict(), indent=2)
    print(summary, file=sys.stderr if args.no_capture else sys.stdout)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(summary + "\n")
    if result.status == "timeout":
        return 124
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
