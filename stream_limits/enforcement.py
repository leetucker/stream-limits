"""Explicit Linux cgroup memory enforcement; no numerical dependencies.

The caller supplies an existing delegated parent. This module never invokes
sudo, mounts controllers, or changes limits on a pre-existing group.
"""

import math
import os
import signal
import sys
import time
from pathlib import Path


class UnsupportedEnforcement(RuntimeError):
    """The requested hard-limit backend is unavailable; no fallback was used."""


def cgroup_capability(parent=None):
    path = Path(parent) if parent else Path("/sys/fs/cgroup/unified")
    if not path.exists():
        path = Path("/sys/fs/cgroup")
    return {
        "path": str(path),
        "v2": (path / "cgroup.controllers").exists(),
        "v1": (path / "memory.limit_in_bytes").exists(),
        "writable": os.access(path, os.W_OK),
        "memory_controller": (path / "memory.max").exists()
        or (path / "memory.limit_in_bytes").exists(),
        "mounts": [
            line
            for line in Path("/proc/mounts").read_text().splitlines()
            if " cgroup" in line
        ]
        if Path("/proc/mounts").exists()
        else [],
        "note": "No cgroup is created unless --cgroup-parent is supplied explicitly.",
    }


class MemoryCgroup:
    """Own one child group; the supervising driver stays outside its limit."""

    def __init__(self, parent, limit):
        if sys.platform != "linux":
            raise UnsupportedEnforcement("cgroup memory enforcement requires Linux")
        if isinstance(limit, bool) or not math.isfinite(limit) or limit < 1:
            raise ValueError("memory limit must be finite and positive")
        parent = Path(parent)
        if (parent / "memory.limit_in_bytes").exists():
            self.version = 1
        elif (parent / "memory.max").exists() or (
            parent / "cgroup.controllers"
        ).exists():
            self.version = 2
        else:
            raise ValueError("parent has no supported memory controller")
        # File names/readback in an ordinary directory do not prove that a
        # kernel controller is present. Verify the actual filesystem first.
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        statfs = libc.statfs
        statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        statfs.restype = ctypes.c_int
        data = ctypes.create_string_buffer(256)
        if statfs(os.fsencode(parent), data) != 0:
            raise OSError(
                ctypes.get_errno(), "cannot inspect cgroup filesystem", str(parent)
            )
        magic = ctypes.c_long.from_buffer(data).value
        expected_magic = 0x27E0EB if self.version == 1 else 0x63677270
        if magic != expected_magic:
            raise UnsupportedEnforcement(
                "parent is not on a genuine kernel memory cgroup filesystem"
            )
        self.limit = int(limit)
        self.path = Path(parent) / f"stream-limits-{os.getpid()}-{time.time_ns()}"
        self.path.mkdir()
        try:
            if self.version == 2:
                (self.path / "memory.max").write_text(str(self.limit))
                (self.path / "memory.swap.max").write_text("0")
                (self.path / "memory.oom.group").write_text("1")
            else:
                (self.path / "memory.limit_in_bytes").write_text(str(self.limit))
                (self.path / "memory.memsw.limit_in_bytes").write_text(str(self.limit))
                (self.path / "memory.swappiness").write_text("0")
                (self.path / "memory.oom_control").write_text("0")
            metrics = self.metrics()
            actual = metrics["memory_max_bytes"]
            if self.version == 1 and (
                metrics["memory_memsw_max_bytes"] != actual
                or metrics["swappiness"] != 0
            ):
                raise RuntimeError(
                    "kernel swap-policy readback differs from requested policy"
                )
            if self.version == 2 and metrics["swap_max_bytes"] != 0:
                raise RuntimeError(
                    "kernel swap-policy readback differs from requested policy"
                )
            if not 0 < actual <= self.limit or self.limit - actual >= os.sysconf(
                "SC_PAGE_SIZE"
            ):
                raise RuntimeError(
                    "kernel memory-limit readback differs from requested cap"
                )
        except BaseException:
            self.path.rmdir()
            raise

    def command(self, command):
        return [
            sys.executable,
            "-m",
            "stream_limits._exec_cgroup",
            str(self.path),
            "--",
            *map(str, command),
        ]

    def _effective_limit(self):
        best = self.limit
        filename = "memory.limit_in_bytes" if self.version == 1 else "memory.max"
        for path in [self.path, *self.path.parents]:
            control = path / filename
            if not control.exists():
                break
            value = control.read_text().strip()
            if value.isdigit() and 0 < int(value) < (1 << 60):
                best = min(best, int(value))
        return best

    def metrics(self):
        if self.version == 1:
            oom = dict(
                line.split()
                for line in (self.path / "memory.oom_control").read_text().splitlines()
            )
            return {
                "version": 1,
                "effective_memory_max_bytes": self._effective_limit(),
                "memory_max_bytes": int(
                    (self.path / "memory.limit_in_bytes").read_text()
                ),
                "memory_memsw_max_bytes": int(
                    (self.path / "memory.memsw.limit_in_bytes").read_text()
                ),
                "swappiness": int((self.path / "memory.swappiness").read_text()),
                "memory_peak_bytes": int(
                    (self.path / "memory.max_usage_in_bytes").read_text()
                ),
                "events": {
                    "oom_kill": int(oom.get("oom_kill", 0)),
                    "limit_failures": int((self.path / "memory.failcnt").read_text()),
                },
                "swap_policy": "memory+swap limit equals memory limit; swappiness=0",
            }
        values = {
            "version": 2,
            "effective_memory_max_bytes": self._effective_limit(),
            "memory_max_bytes": int((self.path / "memory.max").read_text()),
            "swap_max_bytes": int((self.path / "memory.swap.max").read_text()),
            "events": {
                k: int(v)
                for k, v in (
                    line.split()
                    for line in (self.path / "memory.events").read_text().splitlines()
                )
            },
        }
        peak = self.path / "memory.peak"
        values["memory_peak_bytes"] = int(peak.read_text()) if peak.exists() else None
        return values

    def close(self):
        # A command can leave background descendants. Only this owned child
        # group is touched, never the delegated parent or a system service.
        deadline = time.monotonic() + 3
        while True:
            pids = (self.path / "cgroup.procs").read_text().split()
            if not pids:
                break
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                raise RuntimeError("processes remain in the owned memory cgroup")
            time.sleep(0.01)
        self.path.rmdir()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
