"""Variables the user has set are left alone unless `override=True`; loaded
libraries are only lowered; and the wider set of libraries is covered."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

import stream_limits as sl

has_affinity = pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="needs os.sched_getaffinity")

SCRUB_PREFIXES = ("STREAM_LIMITS", "POLARS_", "NUMEXPR_", "TF_NUM_", "BLIS_", "RAYON_", "LOKY_",
                   "JULIA_", "OMP_", "OPENBLAS_", "MKL_", "VECLIB_", "NUMBA_")
SCRUB = ("PBS_NCPUS", "NCPUS", "PBS_NUM_PPN", "PBS_NP", "SLURM_CPUS_PER_TASK",
         "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "PBS_JOBID")


def _run(code, **env):
    e = {k: v for k, v in os.environ.items()
         if not (k.startswith(SCRUB_PREFIXES) or k in SCRUB)}
    e.update(env)
    return subprocess.run([sys.executable, "-W", "always", "-c", code], env=e,
                          capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------- the variable list
def test_the_wider_set_of_libraries_gets_its_variable():
    env: dict = {}
    r = sl.limit_threads(3, environ=env)
    for name in ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
                 "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
                 "RAYON_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                 "LOKY_MAX_CPU_COUNT", "JULIA_NUM_THREADS"):
        assert env[name] == "3", name
        assert name in r["set_env"]
    assert r["kept"] == {}


# ------------------------------------------------------------------ override
def test_variables_the_user_set_are_kept_by_default():
    env = {"OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "2"}
    r = sl.limit_threads(4, environ=env)
    assert env["OMP_NUM_THREADS"] == "1" and env["POLARS_MAX_THREADS"] == "2"
    assert r["kept"] == {"OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "2"}
    assert "OMP_NUM_THREADS" not in r["set_env"] and env["OPENBLAS_NUM_THREADS"] == "4"


def test_override_replaces_them():
    env = {"OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "2"}
    r = sl.limit_threads(4, override=True, environ=env)
    assert env["OMP_NUM_THREADS"] == "4" and env["POLARS_MAX_THREADS"] == "4"
    assert r["kept"] == {} and "OMP_NUM_THREADS" in r["set_env"]


def test_an_empty_variable_counts_as_unset():
    env = {"OMP_NUM_THREADS": " "}
    sl.limit_threads(4, environ=env)
    assert env["OMP_NUM_THREADS"] == "4"


def test_a_kept_value_above_the_cap_is_warned_about():
    env = {"OMP_NUM_THREADS": "64", "JULIA_NUM_THREADS": "auto", "RAYON_NUM_THREADS": "2"}
    with pytest.warns(RuntimeWarning, match="OMP_NUM_THREADS=64.*override=True") as w:
        sl.limit_threads(4, environ=env)
    text = str(w[0].message)
    assert "JULIA_NUM_THREADS" not in text and "RAYON_NUM_THREADS" not in text   # not above the cap
    assert env["OMP_NUM_THREADS"] == "64"
    import warnings
    with warnings.catch_warnings():
        # only this warning: others (Polars already imported, in a process that has loaded it) are not the point
        warnings.filterwarnings("error", message=".*left alone.*")
        sl.limit_threads(4, override=True, environ={"OMP_NUM_THREADS": "64"})   # no warning


def test_the_warning_points_at_the_caller():
    out = _run("import stream_limits\nstream_limits.limit_threads(2)", OMP_NUM_THREADS="64")
    assert "<string>:2" in out.stderr and "OMP_NUM_THREADS=64" in out.stderr, out.stderr


def test_apply_scheduler_defaults_also_keeps_the_users_variables():
    out = _run("import os, stream_limits as s; r = s.apply_scheduler_defaults();"
               "print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'], r['threads']['kept'])",
               PBS_NCPUS="4", OMP_NUM_THREADS="2")
    assert out.stdout.split()[:2] == ["2", "4"], out.stderr


def test_set_limits_passes_override_through():
    a = _run("import os, stream_limits as s; s.set_limits(threads=3); print(os.environ['OMP_NUM_THREADS'])",
             OMP_NUM_THREADS="1")
    b = _run("import os, stream_limits as s; s.set_limits(threads=3, override=True);"
             "print(os.environ['OMP_NUM_THREADS'])", OMP_NUM_THREADS="1")
    assert a.stdout.strip() == "1" and b.stdout.strip() == "3", (a.stderr, b.stderr)


def test_the_stream_limits_variable_always_follows_an_explicit_argument():
    out = _run("import os, stream_limits as s; s.set_limits(threads=5);"
               "print(os.environ['STREAM_LIMITS_MAX_THREADS'], s.detect_cpus())",
               STREAM_LIMITS_MAX_THREADS="2")
    assert out.stdout.split() == ["5", "5"], out.stderr


# ---------------------------------------------------- loaded libraries: lower only
@has_affinity
def test_numba_is_only_lowered_unless_override():
    out = _run("import numba, stream_limits as s; numba.set_num_threads(2);"
               "s.limit_threads(6); a = numba.get_num_threads();"
               "s.limit_threads(6, override=True); b = numba.get_num_threads(); print(a, b)")
    a, b = map(int, out.stdout.split())
    assert a == 2                                # the user's lower setting stands
    assert b == min(6, len(os.sched_getaffinity(0)))   # override sets it (up to the launch maximum)


def test_blas_is_only_lowered_unless_override():
    out = _run(
        "import numpy, threadpoolctl as t, stream_limits as s\n"
        "found = [d for d in t.threadpool_info() if d['user_api'] == 'blas']\n"
        "if not found: print('none'); raise SystemExit\n"
        "blas = lambda: max(d['num_threads'] for d in t.threadpool_info() if d['user_api'] == 'blas')\n"
        "for lib in t.ThreadpoolController().lib_controllers:\n"
        "    if lib.user_api == 'blas': lib.set_num_threads(1)\n"
        "s.limit_threads(4); a = blas()\n"
        "s.limit_threads(4, override=True); b = blas(); print(a, b)")
    if out.stdout.strip() == "none":
        pytest.skip("threadpoolctl sees no BLAS here (Apple's Accelerate, for one)")
    a, b = map(int, out.stdout.split())
    assert a == 1 and b == 4, out.stderr


def test_the_blas_is_not_reported_lowered_when_there_is_none(monkeypatch):
    import threadpoolctl
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setattr(threadpoolctl, "ThreadpoolController",
                        lambda: types.SimpleNamespace(lib_controllers=[]))
    assert "blas" not in sl.limit_threads(2, environ={})["lowered"]


def test_pyarrow_is_only_lowered_unless_override():
    out = _run("import pyarrow as pa, stream_limits as s; pa.set_cpu_count(1);"
               "s.limit_threads(3); a = pa.cpu_count();"
               "s.limit_threads(3, override=True); print(a, pa.cpu_count())")
    assert out.stdout.split() == ["1", "3"], out.stderr


def _has(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


@pytest.mark.skipif(not _has("numexpr"), reason="numexpr is not installed")
def test_numexpr_is_lowered_never_raised_and_gets_its_variables():
    out = _run("import numexpr as ne, stream_limits as s; before = ne.get_num_threads();"
               "r = s.limit_threads(2); a = ne.get_num_threads();"
               "ne.set_num_threads(1); s.limit_threads(2); b = ne.get_num_threads();"
               "s.limit_threads(2, override=True); c = ne.get_num_threads();"
               "print(before, a, b, c, 'numexpr' in r['lowered'])")
    before, a, b, c, listed = out.stdout.split()[-5:]
    assert int(before) >= 2 and (a, b, c, listed) == ("2", "1", "2", "True"), out.stderr
    pre = _run("import stream_limits as s; s.limit_threads(2); import numexpr as ne;"
               "print(ne.get_num_threads())")
    assert pre.stdout.strip().splitlines()[-1] == "2", pre.stderr   # the variable, before import


def test_torch_is_lowered_through_its_api(monkeypatch):
    # PyTorch is not a dependency; this checks the calls made, against a stand-in with its API
    calls = []
    fake = types.SimpleNamespace(get_num_threads=lambda: 8,
                                 set_num_threads=lambda k: calls.append(k))
    monkeypatch.setitem(sys.modules, "torch", fake)
    r = sl.limit_threads(3, environ={})
    assert calls == [3] and "torch" in r["lowered"]
    calls.clear()
    fake.get_num_threads = lambda: 2
    sl.limit_threads(3, environ={})
    assert calls == [2]                                   # only lowered
    calls.clear()
    sl.limit_threads(3, override=True, environ={})
    assert calls == [3]


def test_a_library_that_refuses_is_reported_not_raised(monkeypatch):
    def boom(k):
        raise RuntimeError("cannot change after parallel work has started")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        get_num_threads=lambda: 8, set_num_threads=boom))
    r = sl.limit_threads(3, environ={})
    assert "torch" in r["too_late"] and "torch" not in r["lowered"]


# --------------------------------------------------------------- the memory side
@pytest.mark.skipif(not sl.polars_spills(), reason="needs Polars 2.0 or later")
def test_limit_memory_override_replaces_a_preset_budget():
    kept = _run("import os, stream_limits as s; r = s.limit_memory(10);"
                "print(r['skipped'], os.environ['POLARS_OOC_MEMORY_BUDGET_MB'])",
                POLARS_OOC_MEMORY_BUDGET_MB="123")
    assert "already set" in kept.stdout and kept.stdout.split()[-1] == "123", kept.stderr
    over = _run("import os, stream_limits as s; r = s.limit_memory(10, override=True);"
                "print(r['budget_mb'], os.environ['POLARS_OOC_MEMORY_BUDGET_MB'],"
                "'POLARS_OOC_MEMORY_BUDGET_FRACTION' in os.environ)",
                POLARS_OOC_MEMORY_BUDGET_MB="123", POLARS_OOC_MEMORY_BUDGET_FRACTION="0.9")
    assert over.stdout.split() == ["4250", "4250", "False"], over.stderr


@pytest.mark.skipif(not sl.polars_spills(), reason="needs Polars 2.0 or later")
def test_a_preset_spill_dir_is_kept_unless_override(tmp_path):
    code = (f"import os, stream_limits as s; s.limit_memory(10, spill_dir={str(tmp_path)!r}"
            "{extra}); print(os.environ['POLARS_OOC_SPILL_DIR'])")
    a = _run(code.format(extra=""), POLARS_OOC_SPILL_DIR="/somewhere/else")
    b = _run(code.format(extra=", override=True"), POLARS_OOC_SPILL_DIR="/somewhere/else")
    assert a.stdout.strip() == "/somewhere/else" and b.stdout.strip() == str(tmp_path), (a.stderr, b.stderr)


# --- set_openmp_wait_policy ---

def test_wait_policy_sets_the_variable_and_respects_the_user():
    env = {}
    assert sl.set_openmp_wait_policy(environ=env) == {"policy": "passive", "set": True, "late": None}
    assert env["OMP_WAIT_POLICY"] == "passive"
    env = {"OMP_WAIT_POLICY": "ACTIVE"}
    assert sl.set_openmp_wait_policy(environ=env) == {"policy": "active", "set": False, "late": None}
    assert env["OMP_WAIT_POLICY"] == "ACTIVE"
    assert sl.set_openmp_wait_policy(environ=env, override=True)["set"] is True
    assert env["OMP_WAIT_POLICY"] == "passive"
    with pytest.raises(ValueError):
        sl.set_openmp_wait_policy("sleepy", environ={})


def test_importing_the_module_leaves_the_policy_alone():
    r = _run("import os, stream_limits; print(os.environ.get('OMP_WAIT_POLICY'))")
    assert r.stdout.strip() == "None"
    r = _run("import os, stream_limits; stream_limits.apply_scheduler_defaults(); "
             "print(os.environ.get('OMP_WAIT_POLICY'))", NCPUS="2")
    assert r.stdout.strip() == "None"


NUMBA_PROBE = """
import numpy as np, sys
{before}
import numba
from numba import njit, prange
import stream_limits
print(stream_limits.set_openmp_wait_policy()["late"])
@njit(parallel=True)
def f(a):
    s = 0.0
    for i in prange(a.size):
        s += a[i]
    return s
f(np.ones(100))
{after}
"""


def test_wait_policy_takes_effect_when_numba_is_imported_first():
    # numba loads its OpenMP runtime at the first parallel call, which is after
    r = _run(NUMBA_PROBE.format(before="", after="print(numba.threading_layer())"),
             OMP_DISPLAY_ENV="verbose")
    if "omp" not in r.stdout.split():
        pytest.skip("numba's OpenMP threading layer is not in use")
    # "late" is only detectable on Linux (it reads /proc/self/maps)
    assert r.stdout.split()[0] == ("False" if sys.platform.startswith("linux") else "None")
    assert "OMP_WAIT_POLICY = 'PASSIVE'" in r.stderr


def test_wait_policy_reports_a_runtime_that_is_already_loaded():
    r = _run(NUMBA_PROBE.format(before="", after="") +
             "\nimport stream_limits as s\nprint(s.set_openmp_wait_policy(override=True)['late'])")
    if not sys.platform.startswith("linux"):
        pytest.skip("detection needs /proc/self/maps")
    lines = r.stdout.split()
    if lines[-1] == "False":
        pytest.skip("numba's OpenMP layer is not in use")
    assert lines[-1] == "True"
