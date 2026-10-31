"""Limits on the threads and memory a Python process may use, for jobs on a scheduler.

Polars, numba, BLAS and pyarrow each size their thread pools separately, and
each defaults to every core of the *node*, not of the job; memory guards that
look at the machine's physical memory see the node's too. This module finds
the allocation a job has (PBS, Slurm, cgroups, or your own setting), caps each
library at it as far as the library allows, and measures the memory a block of
code used.

It depends on nothing outside the standard library and imports none of the
libraries it limits, so it can run *before* they are imported, which matters:
Polars reads ``POLARS_MAX_THREADS`` (and its out-of-core memory budget) once,
at import, and nothing can change them afterwards. Put it first in a script::

    import stream_limits
    stream_limits.set_limits()           # the job's allocation; or threads=8, memory_gb=16

    import polars as pl                  # now capped
    import numpy as np

`set_limits` is `limit_threads` and `limit_memory` in one call; use those for
the options the one call leaves out.

The packages that use it (hdfe-stream, logit-stream, dpd-stream) call
`apply_scheduler_defaults` when imported, which does both only if it finds a
limit; on a laptop it does nothing. Importing *this* module changes nothing.

Where the limits come from, first match wins
    threads  the argument; ``STREAM_LIMITS_MAX_THREADS``; the variables named
             in ``STREAM_LIMITS_CPU_VARS``; ``NCPUS``, ``PBS_NCPUS`` (PBS Pro),
             ``PBS_NUM_PPN``, ``PBS_NP`` (Torque); ``SLURM_CPUS_PER_TASK``,
             ``SLURM_CPUS_ON_NODE``; the cgroup CPU quota; PBS's own record of
             the job, ``qstat -f $PBS_JOBID`` (``Resource_List.ncpus``, else
             ``select``, ``nodes``, ``procs``); the process's CPU affinity;
             ``os.cpu_count()``
    memory   ``STREAM_LIMITS_MEMORY_GB``; the variables named in
             ``STREAM_LIMITS_MEMORY_VARS``; the cgroup memory limit (the lowest
             one on the path to the root); ``qstat -f $PBS_JOBID``
             (``Resource_List.mem``, else ``vmem``), for sites that set no
             memory variable and enforce no cgroup; ``SLURM_MEM_PER_NODE``, or
             ``SLURM_MEM_PER_CPU`` times the CPUs; the machine's physical memory

    PBS Pro exports ``NCPUS`` to a job and no memory variable; Torque exports
    ``PBS_NUM_PPN`` and ``PBS_NP``. Sites differ beyond that, so any variable
    that holds a CPU count or a size can be named in the two ``..._VARS``
    settings.

What is limited
    Polars, numba, BLAS (OpenBLAS, MKL, BLIS, Accelerate; NumPy and SciPy),
    OpenMP (so LightGBM, XGBoost and scikit-learn's compiled code), pyarrow,
    numexpr, rayon-based Rust libraries, TensorFlow, joblib's workers and
    Julia: by environment variable, before they are imported. Libraries already
    imported are also lowered at run time where they allow it (numba, BLAS,
    pyarrow, numexpr, PyTorch). Variables you have set yourself are left alone,
    and loaded libraries only lowered, unless ``override=True``. Libraries with
    only per-object settings (DuckDB, Dask, Ray, JAX) are in the documentation's
    "Other libraries".

Other environment variables
    STREAM_LIMITS_MEMORY_RESERVE   share of the memory limit `MemoryBudget`
                                   keeps back (default 0.15)
    STREAM_LIMITS_POLARS_SPILL     ``0`` stops `limit_memory` setting Polars'
                                   out-of-core budget
    STREAM_LIMITS_POLARS_FRACTION  the budget's share of the memory limit
                                   (default 0.5)
    STREAM_LIMITS_SPILL_DIR        where Polars spills (default: its temp dir)
    STREAM_LIMITS_QSTAT            ``0`` stops the ``qstat`` lookups
    STREAM_LIMITS_CPU_VARS         comma-separated names of variables holding a
                                   CPU count, checked before the built-in ones
    STREAM_LIMITS_MEMORY_VARS      the same for memory; a value is a PBS size
                                   (``16gb``, ``512mb``; a bare number is bytes)

``python -m stream_limits`` prints what was found, which is the quickest way
to check a scheduler's settings from inside a job.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterator, Mapping

__version__ = "0.1.0"

__all__ = [
    "set_limits", "limit_threads", "limit_memory", "apply_scheduler_defaults", "thread_scope",
    "detect_cpus", "detect_memory_gb", "describe", "MemoryBudget",
    "peak_rss_gb", "reset_peak_rss", "current_rss_gb",
]

# set to the thread count before the libraries are imported; a variable the user
# has already set is left alone unless `override` is given
_THREAD_ENV = (
    "POLARS_MAX_THREADS",       # Polars' pool, fixed at import
    "NUMBA_NUM_THREADS",        # numba's pool; it can be lowered later, never raised
    "OMP_NUM_THREADS",          # OpenMP, and so most BLAS builds, LightGBM, XGBoost, ...
    "OMP_THREAD_LIMIT",         # OpenMP's hard cap, which nested parallel regions respect
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # Accelerate on macOS
    "NUMEXPR_NUM_THREADS",      # numexpr (and pandas' eval) ...
    "NUMEXPR_MAX_THREADS",      # ... whose pool is also bounded by this, 64 by default
    "RAYON_NUM_THREADS",        # Rust libraries on rayon's global pool (not Polars, which has its own)
    "TF_NUM_INTRAOP_THREADS",   # TensorFlow; fixed when it initializes
    "TF_NUM_INTEROP_THREADS",
    "LOKY_MAX_CPU_COUNT",       # joblib's worker pool (and so scikit-learn's n_jobs=-1)
    "JULIA_NUM_THREADS",        # Julia started from Python
)
_CPU_ENV = ("STREAM_LIMITS_MAX_THREADS", "NCPUS", "PBS_NCPUS", "PBS_NUM_PPN", "PBS_NP",
            "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
_UNLIMITED = 1 << 60            # cgroup v1 reports "no limit" as a huge number
_CGROUP = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")

DEFAULT_RESERVE = 0.15          # share of the memory limit MemoryBudget keeps back
_POLARS_BUDGET_VARS = ("POLARS_OOC_MEMORY_BUDGET_MB", "POLARS_OOC_MEMORY_BUDGET_FRACTION")
_READER_FLOOR_GB = 1.5          # what Polars' Parquet readers hold beyond its budget


# ------------------------------------------------------------ detection
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _cgroup_dirs(controller: str, root: Path, proc: Path) -> list[Path]:
    """The directories of this process's cgroup for a controller (``cpu`` or
    ``memory``) and of every parent up to the root: the effective limit is
    the lowest along the path, because a scheduler usually limits a parent
    (the job) and not the leaf the process sits in. Empty if there is none."""
    text = _read(proc)
    if not text:
        return []
    v2 = (root / "cgroup.controllers").exists()
    rel = None
    for line in text.splitlines():
        _, ctrls, path = (line.split(":", 2) + ["", ""])[:3]
        if v2 and ctrls == "":
            rel = path
        elif not v2 and controller in ctrls.split(","):
            rel = path
    if rel is None:
        return []
    base = root if v2 else root / controller
    if not v2 and not base.exists():                    # "cpu,cpuacct" is one mount
        base = next((p for p in root.glob(f"*{controller}*") if p.is_dir()), base)
    parts = [p for p in rel.split("/") if p]
    dirs = [base.joinpath(*parts[:i]) for i in range(len(parts), -1, -1)]
    existing = [d for d in dirs if d.is_dir()]
    # inside a container the path may be "/" or not exist under this root
    return existing or [base]


def _cgroup_cpu_limit(root: Path = _CGROUP, proc: Path = _PROC_CGROUP) -> float | None:
    """The CPU quota in CPUs (quota / period), or None if there is none."""
    best = None
    for d in _cgroup_dirs("cpu", root, proc):
        quota = period = None
        text = _read(d / "cpu.max")                       # v2: "200000 100000" or "max 100000"
        if text:
            q, _, p = text.partition(" ")
            if q != "max" and q.isdigit() and p.isdigit():
                quota, period = int(q), int(p)
        else:                                             # v1
            q, p = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
            if q and p and q.lstrip("-").isdigit() and p.isdigit() and int(q) > 0:
                quota, period = int(q), int(p)
        if quota and period:
            cpus = quota / period
            best = cpus if best is None else min(best, cpus)
    return best


def _cgroup_memory_limit(root: Path = _CGROUP, proc: Path = _PROC_CGROUP) -> int | None:
    """The cgroup memory limit in bytes (the lowest on the path), or None."""
    best = None
    for d in _cgroup_dirs("memory", root, proc):
        text = _read(d / "memory.max") or _read(d / "memory.limit_in_bytes")
        if text and text.isdigit() and 0 < int(text) < _UNLIMITED:
            best = int(text) if best is None else min(best, int(text))
    return best


def _int_env(environ: Mapping[str, str], name: str) -> int | None:
    try:
        v = int(float(environ[name]))
    except (KeyError, ValueError):
        return None
    return v if v > 0 else None


def _machine_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def _cpu_limit(environ, root, proc) -> tuple[int, str] | None:
    """The CPU count a scheduler, container or the user imposes, with where it
    came from; None when nothing limits the machine."""
    for name in (_CPU_ENV[0], *_named(environ, "STREAM_LIMITS_CPU_VARS"), *_CPU_ENV[1:]):
        v = _int_env(environ, name)
        if v:
            return v, name
    q = _cgroup_cpu_limit(root, proc)
    if q is not None:
        return max(1, math.ceil(q)), "cgroup cpu quota"
    jobid = environ.get("PBS_JOBID")
    if jobid and _qstat_enabled(environ):
        found = _qstat_cpus(jobid)
        if found:                       # a total over chunks must not exceed this machine
            return min(found[0], _machine_cpus()), found[1]
    try:
        n_aff, n_all = len(os.sched_getaffinity(0)), os.cpu_count() or 0
    except AttributeError:                                # macOS, Windows
        return None
    if 0 < n_aff < n_all:
        return n_aff, "cpu affinity"
    return None


def detect_cpus(environ: Mapping[str, str] | None = None, *,
                cgroup_root: Path = _CGROUP, proc_cgroup: Path = _PROC_CGROUP) -> int:
    """The number of CPUs this process may use: the limit a scheduler,
    container or ``STREAM_LIMITS_MAX_THREADS`` sets (see the module docstring
    for the order), otherwise the machine's."""
    environ = os.environ if environ is None else environ
    lim = _cpu_limit(environ, Path(cgroup_root), Path(proc_cgroup))
    return lim[0] if lim else _machine_cpus()


_SIZE_UNITS = {"": 1, "b": 1, "w": 8, "kb": 1024, "kw": 8 * 1024, "mb": 1024**2,
               "mw": 8 * 1024**2, "gb": 1024**3, "gw": 8 * 1024**3, "tb": 1024**4,
               "tw": 8 * 1024**4, "pb": 1024**5, "pw": 8 * 1024**5}
_JOBID = re.compile(r"^[A-Za-z0-9_.\-\[\]@:]+$")
_QSTAT_TIMEOUT = 10.0


def _parse_pbs_size(text: str) -> float | None:
    """A PBS size (``16gb``, ``512mb``, ``2tb``, ``1000000``; no suffix is
    bytes, ``w`` is words of 8 bytes) in bytes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", text)
    if not m or m.group(2).lower() not in _SIZE_UNITS:
        return None
    return float(m.group(1)) * _SIZE_UNITS[m.group(2).lower()]


def _qstat_enabled(environ) -> bool:
    return environ.get("STREAM_LIMITS_QSTAT", "1").strip().lower() not in ("0", "false", "no")


@functools.lru_cache(maxsize=8)
def _qstat_resources(jobid: str) -> dict | None:
    """The ``Resource_List.*`` attributes PBS records for a job, as strings.
    Runs ``qstat -f <jobid>`` once per job (cached), without a shell and with a
    timeout; None if there is no ``qstat``, it fails, or the id is not a job id."""
    if not _JOBID.match(jobid):
        return None
    try:
        out = subprocess.run(["qstat", "-f", jobid], capture_output=True, text=True,
                             timeout=_QSTAT_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # attributes are indented four spaces; a long value wraps onto lines that start with a tab
    text = re.sub(r"\n\t", "", out.stdout)
    found = dict(re.findall(r"^\s*Resource_List\.(\w+)\s*=\s*(\S+)\s*$", text, re.M))
    return found or None


def _qstat_memory(jobid: str) -> tuple[float, str] | None:
    """The memory PBS records for a job, in GB, and which field it came from:
    ``Resource_List.mem``, else ``vmem`` (virtual memory, a looser limit). For a
    job with several chunks PBS reports the total over all of them."""
    res = _qstat_resources(jobid)
    for field in ("mem", "vmem"):
        size = _parse_pbs_size(res[field]) if res and field in res else None
        if size:
            return size / 1e9, f"qstat Resource_List.{field}"
    return None


def _chunk_cpus(spec: str, key: str) -> int | None:
    """CPUs of the first chunk of a PBS ``select`` (``2:ncpus=8:mem=4gb+...``) or
    Torque ``nodes`` (``2:ppn=4+...``) request: the chunk count times `key`."""
    first = spec.split("+")[0].split(":")
    count = int(first[0]) if first and first[0].isdigit() else 1
    for tok in first:
        if tok.startswith(key + "="):
            try:
                return count * int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return count if key == "ppn" and first and first[0].isdigit() else None


def _qstat_cpus(jobid: str) -> tuple[int, str] | None:
    """The CPUs PBS records for a job, and which field: ``Resource_List.ncpus``
    (PBS Pro), else the first chunk of ``select`` (PBS Pro) or ``nodes`` (Torque),
    else ``procs``. For a job with several chunks ``ncpus`` is the total over all
    of them, and the caller caps it at the machine's CPUs."""
    res = _qstat_resources(jobid)
    if not res:
        return None
    v = res.get("ncpus")
    if v and v.isdigit() and int(v) > 0:
        return int(v), "qstat Resource_List.ncpus"
    for field, key in (("select", "ncpus"), ("nodes", "ppn")):
        if field in res:
            n = _chunk_cpus(res[field], key)
            if n:
                return n, f"qstat Resource_List.{field}"
    v = res.get("procs")
    if v and v.isdigit() and int(v) > 0:
        return int(v), "qstat Resource_List.procs"
    return None


def _named(environ, setting: str) -> list[str]:
    return [n.strip() for n in environ.get(setting, "").split(",") if n.strip()]


def _memory(environ, root, proc) -> tuple[float, str]:
    v = environ.get("STREAM_LIMITS_MEMORY_GB")
    try:
        if v and float(v) > 0:
            return float(v), "STREAM_LIMITS_MEMORY_GB"
    except ValueError:
        pass
    for name in _named(environ, "STREAM_LIMITS_MEMORY_VARS"):
        size = _parse_pbs_size(environ.get(name, ""))
        if size:
            return size / 1e9, name
    cg = _cgroup_memory_limit(root, proc)
    if cg:
        return cg / 1e9, "cgroup memory limit"
    jobid = environ.get("PBS_JOBID")
    if jobid and _qstat_enabled(environ):
        q = _qstat_memory(jobid)
        if q:
            return q
    v = _int_env(environ, "SLURM_MEM_PER_NODE")          # megabytes
    if v:
        return v * 1048576 / 1e9, "SLURM_MEM_PER_NODE"
    v = _int_env(environ, "SLURM_MEM_PER_CPU")           # megabytes per CPU
    cpus = _int_env(environ, "SLURM_CPUS_ON_NODE") or _int_env(environ, "SLURM_CPUS_PER_TASK")
    if v and cpus:
        return v * cpus * 1048576 / 1e9, "SLURM_MEM_PER_CPU x CPUs"
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, "physical memory"
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                        ("virt_total", ctypes.c_ulonglong), ("virt_avail", ctypes.c_ulonglong),
                        ("ext", ctypes.c_ulonglong)]
        status = _Status(length=ctypes.sizeof(_Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 1e9, "physical memory"
    return 16.0, "default"


def detect_memory_gb(environ: Mapping[str, str] | None = None, *,
                     cgroup_root: Path = _CGROUP, proc_cgroup: Path = _PROC_CGROUP) -> float:
    """The memory in GB this process may use: ``STREAM_LIMITS_MEMORY_GB``, the
    cgroup limit a scheduler sets for the job, or else the machine's physical
    memory."""
    environ = os.environ if environ is None else environ
    return _memory(environ, Path(cgroup_root), Path(proc_cgroup))[0]


def describe(environ: Mapping[str, str] | None = None) -> dict:
    """What was found, and what the libraries are doing now: for a log line,
    a bug report, or ``python -m stream_limits``."""
    environ = os.environ if environ is None else environ
    lim = _cpu_limit(environ, _CGROUP, _PROC_CGROUP)
    mem, mem_src = _memory(environ, _CGROUP, _PROC_CGROUP)
    out = {"cpus": detect_cpus(environ),
           "cpus_source": lim[1] if lim else "machine (no limit found)",
           "memory_gb": round(mem, 2), "memory_source": mem_src,
           "env": {k: environ[k] for k in (
               *_CPU_ENV, *_THREAD_ENV, *_POLARS_BUDGET_VARS, "POLARS_OOC_SPILL_DIR",
               "STREAM_LIMITS_MEMORY_GB", "STREAM_LIMITS_MEMORY_RESERVE", "STREAM_LIMITS_CPU_VARS",
               "STREAM_LIMITS_MEMORY_VARS", "PBS_JOBID", "SLURM_MEM_PER_NODE",
               "SLURM_MEM_PER_CPU") if k in environ}}
    pl = sys.modules.get("polars")
    if pl is not None:
        out["polars_threads"] = pl.thread_pool_size()
    nb = sys.modules.get("numba")
    if nb is not None:
        out["numba_threads"] = nb.get_num_threads()
    return out


# ------------------------------------------------------------- a budget
class MemoryBudget:
    """How much memory is left to allocate, for guards that must choose between
    a fast path that needs a lot of memory and a slower one that needs less.

    ``headroom_gb()`` is the limit, less the share `reserve` keeps back for
    what the code does not model (allocator slack, transient copies, pages the
    cgroup charges for files), less what the process holds now. A guard asks
    whether *its exact allocation* fits, instead of comparing it with a fixed
    fraction of the machine. Reading the process's memory is one file read.

    Parameters
    ----------
    limit_gb : float, optional
        The memory the process may use. Default: `detect_memory_gb()`.
    reserve : float, optional
        The share of the limit kept back, 0 to 1. Default
        ``$STREAM_LIMITS_MEMORY_RESERVE``, else 0.15: lower it to trust the
        estimates more, raise it for a job that must not be killed.
    """

    def __init__(self, limit_gb: float | None = None, reserve: float | None = None, *,
                 environ: Mapping[str, str] | None = None):
        environ = os.environ if environ is None else environ
        self.limit_gb = detect_memory_gb(environ) if limit_gb is None else float(limit_gb)
        if reserve is None:
            try:
                reserve = float(environ.get("STREAM_LIMITS_MEMORY_RESERVE", DEFAULT_RESERVE))
            except ValueError:
                reserve = DEFAULT_RESERVE
        if not 0 <= reserve < 1:
            raise ValueError("reserve must be at least 0 and below 1")
        if not self.limit_gb > 0:
            raise ValueError("the memory limit must be positive")
        self.reserve = float(reserve)

    @property
    def usable_gb(self) -> float:
        """The limit less the reserve."""
        return self.limit_gb * (1 - self.reserve)

    def held_gb(self) -> float:
        """What the process holds now (0.5 where it cannot be read)."""
        rss = current_rss_gb()
        return 0.5 if rss is None else rss

    def headroom_gb(self, upcoming_gb: float = 0.0) -> float:
        """What can still be allocated, after `upcoming_gb` that will be
        allocated later in the same phase; never negative."""
        return max(0.0, self.usable_gb - self.held_gb() - upcoming_gb)

    def fits(self, need_gb: float, upcoming_gb: float = 0.0, share: float = 1.0) -> bool:
        """Whether an allocation of `need_gb` fits within `share` of the headroom."""
        return need_gb <= share * self.headroom_gb(upcoming_gb)

    def __repr__(self):
        return (f"MemoryBudget(limit_gb={self.limit_gb:.2f}, reserve={self.reserve:.2f}, "
                f"headroom_gb={self.headroom_gb():.2f})")


# ------------------------------------------------------------- limiting
def _above(value: str, n: int) -> bool:
    try:
        return int(value) > n
    except ValueError:                  # "auto", "" ...
        return False


def _ceiling(n: int, current: int | None, override: bool, limit: int | None = None) -> int:
    """The thread count to set in a loaded library: `n`, capped at `limit` (a
    library's launch-time maximum), and unless `override`, never above what it
    already uses, so the cap lowers and never raises."""
    target = n if limit is None else min(n, limit)
    if not override and current is not None:
        target = min(target, current)
    return max(1, int(target))


def limit_threads(n: int | None = None, *, override: bool = False, environ=None,
                  _auto: bool = False, _stack: int = 0) -> dict:
    """Cap the threads that polars, numba, BLAS, pyarrow, numexpr and the other
    libraries named in the module docstring use, as far as each allows.

    Parameters
    ----------
    n : int, optional
        The cap. Default: `detect_cpus()`, the job's allocation.
    override : bool, default False
        By default a thread variable you have already set (``OMP_NUM_THREADS=1``,
        say) is left alone, and loaded libraries are only ever lowered: the
        cap is a ceiling, and a more specific choice of yours stays. A warning
        names any variable you set above the cap. ``override=True`` replaces
        the variables and sets the loaded libraries to exactly `n`.

    Call it before importing polars. Before the libraries are imported it
    sets their environment variables, which is the only way to limit Polars
    (its pool is fixed at import). After they are imported it still lowers
    numba's, BLAS's, pyarrow's, numexpr's and PyTorch's threads, but cannot
    touch Polars', and says so with a warning; a limit Polars would have to
    honor goes in ``STREAM_LIMITS_MAX_THREADS`` (or ``POLARS_MAX_THREADS``)
    in the job script.

    Returns
    -------
    dict
        What it did: ``set_env`` (variables set), ``kept`` (variables you had
        set, with their values, left alone), ``lowered`` (libraries lowered at
        run time) and ``too_late`` (libraries it could not limit).
    """
    environ = os.environ if environ is None else environ
    n = detect_cpus(environ) if n is None else int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    report: dict = {"threads": n, "set_env": [], "kept": {}, "lowered": [], "too_late": []}
    level = (4 if _auto else 2) + _stack
    for name in _THREAD_ENV:
        current = environ.get(name, "").strip()
        if current and not override:
            report["kept"][name] = current
            continue
        environ[name] = str(n)
        report["set_env"].append(name)
    above = {k: v for k, v in report["kept"].items() if _above(v, n)}
    if above:
        warnings.warn(
            f"{', '.join(f'{k}={v}' for k, v in above.items())} {'is' if len(above) == 1 else 'are'} "
            f"set above the thread limit {n} and left alone; pass override=True "
            "(or unset it) to apply the limit.", RuntimeWarning, stacklevel=level)
    pl = sys.modules.get("polars")
    if pl is not None:
        have = pl.thread_pool_size()
        if have > n:
            report["too_late"].append(f"polars ({have} threads)")
            warnings.warn(
                f"Polars was imported before the thread limit and uses {have} threads "
                f"(limit {n}); its pool cannot be resized. Set STREAM_LIMITS_MAX_THREADS={n} "
                "in the job script, or call stream_limits.limit_threads() before "
                "importing polars.", RuntimeWarning, stacklevel=level)
    nb = sys.modules.get("numba")
    if nb is not None:
        try:
            nb.set_num_threads(_ceiling(n, nb.get_num_threads(), override,
                                        nb.config.NUMBA_NUM_THREADS))
            report["lowered"].append("numba")
        except (ValueError, RuntimeError):
            report["too_late"].append("numba")
    if "numpy" in sys.modules:
        try:
            from threadpoolctl import ThreadpoolController
            found = False
            for lib in ThreadpoolController().lib_controllers:
                if lib.user_api == "blas":
                    lib.set_num_threads(_ceiling(n, lib.num_threads, override))
                    found = True
            # threadpoolctl sees OpenBLAS, MKL, BLIS and FlexiBLAS; not Apple's Accelerate,
            # which only VECLIB_MAXIMUM_THREADS (before NumPy is imported) limits
            if found:
                report["lowered"].append("blas")
        except ImportError:
            pass
    pa = sys.modules.get("pyarrow")
    if pa is not None:
        pa.set_cpu_count(_ceiling(n, pa.cpu_count(), override))
        pa.set_io_thread_count(_ceiling(n, pa.io_thread_count(), override))
        report["lowered"].append("pyarrow")
    ne = sys.modules.get("numexpr")
    if ne is not None:
        try:
            # (the module attribute ``nthreads`` is a copy made at import and goes stale)
            current = ne.get_num_threads() if hasattr(ne, "get_num_threads") else None
            ne.set_num_threads(_ceiling(n, current, override))
            report["lowered"].append("numexpr")
        except Exception:                                # noqa: BLE001
            report["too_late"].append("numexpr")
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(_ceiling(n, torch.get_num_threads(), override))
            report["lowered"].append("torch")
        except Exception:                                # noqa: BLE001
            report["too_late"].append("torch")
    return report


def _polars_version() -> tuple[int, ...] | None:
    """The installed Polars' version, read from its metadata (so Polars is not
    imported), or None if it is not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            v = version("polars")
        except PackageNotFoundError:
            return None
    except ImportError:
        return None
    parts = []
    for tok in v.split(".")[:3]:
        digits = "".join(ch for ch in tok if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def polars_spills() -> bool:
    """Whether the installed Polars has the out-of-core memory budget that
    `limit_memory` sets: 2.0 and later (it is an environment variable there,
    undocumented, and read once at import)."""
    v = _polars_version()
    return v is not None and v >= (2, 0)


def limit_memory(limit_gb: float | None = None, *, fraction: float | None = None,
                 spill_dir: str | os.PathLike | None = None, override: bool = False,
                 environ=None, _auto: bool = False, _stack: int = 0) -> dict:
    """Give Polars a memory budget, past which its streaming engine spills to
    disk instead of growing.

    A streaming sort of 1.9 GB of Parquet peaks at 6.2 GB with Polars'
    defaults, a multiple of the data; with a 400 MB budget it peaked at
    1.8 GB, in twice the time. The budget bounds what the engine's operators
    hold, not the whole process: the Parquet readers hold about 1.5 GB more on
    8 threads (less on fewer), and the packages' own arrays are on top.

    Parameters
    ----------
    limit_gb : float, optional
        The memory the process may use. Default: `detect_memory_gb()`.
    fraction : float, optional
        The budget's share of what is left after the readers' floor:
        ``fraction x (limit - 1.5 GB)``, at least 0.25 GB. Default
        ``$STREAM_LIMITS_POLARS_FRACTION``, else 0.5.
    spill_dir : path, optional
        Where Polars spills (``POLARS_OOC_SPILL_DIR``). Default: its own
        temporary directory, which on a cluster should be the job's scratch
        (check ``$TMPDIR``).
    override : bool, default False
        By default a Polars budget or spill directory you have already set in
        the environment is left alone. ``override=True`` replaces them.

    Like the thread limit this must come before Polars is imported (a warning
    says so otherwise). It does nothing when Polars is older than 2.0, when a
    budget is already set in the environment, or when
    ``STREAM_LIMITS_POLARS_SPILL=0``. Polars documents none of this in 2.x;
    the names are those of its binary. Returns what it did."""
    environ = os.environ if environ is None else environ
    report: dict = {"budget_mb": None, "set_env": [], "skipped": None}
    if environ.get("STREAM_LIMITS_POLARS_SPILL", "1").strip().lower() in ("0", "false", "no"):
        report["skipped"] = "STREAM_LIMITS_POLARS_SPILL=0"
        return report
    if not polars_spills():
        report["skipped"] = "Polars is not installed or is older than 2.0"
        return report
    preset = [v for v in _POLARS_BUDGET_VARS if environ.get(v, "").strip()]
    if preset and not override:
        report["skipped"] = "a Polars budget is already set"
        return report
    limit = detect_memory_gb(environ) if limit_gb is None else float(limit_gb)
    if fraction is None:
        try:
            fraction = float(environ.get("STREAM_LIMITS_POLARS_FRACTION", 0.5))
        except ValueError:
            fraction = 0.5
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be above 0 and at most 1")
    budget_mb = int(max(0.25, fraction * (limit - _READER_FLOOR_GB)) * 1000)
    if "polars" in sys.modules:
        report["skipped"] = "Polars was imported first"
        if not _auto or limit_gb is not None:
            warnings.warn(
                "Polars was imported before the memory limit; its out-of-core budget is read "
                "at import and cannot be set now. Call stream_limits.limit_memory() before "
                "importing polars, or set POLARS_OOC_MEMORY_BUDGET_MB in the job script.",
                RuntimeWarning, stacklevel=(4 if _auto else 2) + _stack)
        return report
    for v in _POLARS_BUDGET_VARS:                       # the one budget, not two that disagree
        environ.pop(v, None)
    environ["POLARS_OOC_MEMORY_BUDGET_MB"] = str(budget_mb)
    report["budget_mb"] = budget_mb
    report["set_env"].append("POLARS_OOC_MEMORY_BUDGET_MB")
    spill_dir = spill_dir or environ.get("STREAM_LIMITS_SPILL_DIR")
    if spill_dir and (override or not environ.get("POLARS_OOC_SPILL_DIR", "").strip()):
        environ["POLARS_OOC_SPILL_DIR"] = os.fspath(spill_dir)
        report["set_env"].append("POLARS_OOC_SPILL_DIR")
    return report


def set_limits(threads: int | None = None, memory_gb: float | None = None, *,
               spill_dir: str | os.PathLike | None = None, override: bool = False,
               environ=None) -> dict:
    """Cap the threads and Polars' memory in one call: `limit_threads` and
    `limit_memory`, for most scripts the only line needed, first, before
    Polars is imported.

    A value you give is also recorded in ``STREAM_LIMITS_MAX_THREADS`` or
    ``STREAM_LIMITS_MEMORY_GB``, so `detect_cpus`, `detect_memory_gb` and
    `MemoryBudget` in this process (and in the packages built on this one, and in
    child processes) agree with it. ``set_limits()`` with no arguments does what
    `apply_scheduler_defaults`, the call the libraries make when imported, does.

    Parameters
    ----------
    threads : int, optional
        The thread cap. Given, it is applied (and overrides thread variables
        already set). Omitted, it is the job's CPU allocation if one is found
        (see `detect_cpus`) and nothing otherwise.
    memory_gb : float, optional
        The memory limit in GB. Given, it is applied. Omitted, it is the job's
        allocation if one is found (see `detect_memory_gb`) and nothing
        otherwise: a workstation with no limit gets no Polars budget.
    spill_dir : path, optional
        Where Polars spills; see `limit_memory`.
    override : bool, default False
        Replace thread and Polars variables you have already set, instead of
        leaving them alone (and set loaded libraries to exactly `threads`
        instead of only lowering them); see `limit_threads`.

    Returns
    -------
    dict
        ``{"threads": ..., "memory": ...}``, each the report of `limit_threads`
        or `limit_memory`, left out when there was no limit to apply.
    """
    environ = os.environ if environ is None else environ
    out: dict = {}
    if threads is not None:
        out["threads"] = limit_threads(threads, override=override, environ=environ, _stack=1)
        environ["STREAM_LIMITS_MAX_THREADS"] = str(int(threads))
    else:
        lim = _cpu_limit(environ, _CGROUP, _PROC_CGROUP)
        if lim is not None:
            out["threads"] = limit_threads(lim[0], override=override, environ=environ,
                                           _auto=True, _stack=-1)
    if memory_gb is not None:
        out["memory"] = limit_memory(memory_gb, spill_dir=spill_dir, override=override,
                                     environ=environ, _stack=1)
        environ["STREAM_LIMITS_MEMORY_GB"] = str(float(memory_gb))
    else:
        mem, src = _memory(environ, _CGROUP, _PROC_CGROUP)
        if src not in ("physical memory", "default"):
            out["memory"] = limit_memory(mem, spill_dir=spill_dir, override=override,
                                         environ=environ, _auto=True, _stack=-1)
    return out


def apply_scheduler_defaults(environ=None) -> dict | None:
    """Cap the threads and Polars' memory at the scheduler's allocation, if
    there is one. The estimator packages call this first thing when imported.
    It acts only on limits it finds (a variable, a scheduler's environment,
    a cgroup, a restricted CPU affinity): the thread cap sets only the
    variables the user has not set; with no limit it does nothing, so a
    workstation is untouched. Returns ``{"threads": ..., "memory": ...}``
    (each the report of `limit_threads` / `limit_memory`), or None."""
    environ = os.environ if environ is None else environ
    out: dict = {}
    lim = _cpu_limit(environ, _CGROUP, _PROC_CGROUP)
    if lim is not None:
        out["threads"] = limit_threads(lim[0], environ=environ, _auto=True)
    mem, src = _memory(environ, _CGROUP, _PROC_CGROUP)
    if src not in ("physical memory", "default"):
        out["memory"] = limit_memory(mem, environ=environ, _auto=True)
    return out or None


@contextlib.contextmanager
def thread_scope(n: int | None = None) -> Iterator[None]:
    """Hold numba and BLAS to `n` threads inside the block and restore them
    after. ``None`` (or 0) leaves everything as it is. Polars is not affected:
    its pool is fixed for the process."""
    if not n:
        yield
        return
    import numba
    from threadpoolctl import threadpool_limits
    previous = numba.get_num_threads()
    numba.set_num_threads(max(1, min(int(n), numba.config.NUMBA_NUM_THREADS)))
    try:
        with threadpool_limits(limits=int(n), user_api="blas"):
            yield
    finally:
        numba.set_num_threads(previous)


# ----------------------------------------------------- peak memory used
def _windows_memory() -> tuple[int, int] | None:
    """This process's ``(working set, peak working set)`` in bytes, from the
    Windows API (``GetProcessMemoryInfo``), or None: off Windows, or if the call
    fails. The working set is the physical memory resident in the process, which
    is what ``VmRSS`` is on Linux, and the peak is a high-water mark the system
    keeps."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):          # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE        # 64-bit safe
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                                                     wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                                ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except Exception:                                          # noqa: BLE001
        return None


def reset_peak_rss() -> None:
    """Restart the peak-memory counter at the current usage. Linux only: macOS
    and Windows keep a counter that cannot be reset, so there `peak_rss_gb`
    reports the process's lifetime peak (a second measurement in the same
    process is the largest of both)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def peak_rss_gb() -> float | None:
    """The peak resident memory, in GB, since `reset_peak_rss` (or since the
    process started). The system keeps the counter, so reading it is free and
    nothing is polled: ``VmHWM`` on Linux, ``getrusage`` on macOS, the peak
    working set on Windows. Child processes are not included."""
    win = _windows_memory()
    if win is not None:
        return win[1] / 1e9
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9   # bytes vs KB


def current_rss_gb() -> float | None:
    """The resident memory in GB right now (Linux and Windows; None on macOS,
    where `MemoryBudget` then assumes 0.5 GB)."""
    win = _windows_memory()
    if win is not None:
        return win[0] / 1e9
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    return None
