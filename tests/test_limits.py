"""Detection from the environment and cgroup files, the caps applied to each
library, Polars' memory budget, the memory budget, and peak-memory readings."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import stream_limits as sl

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="needs Linux (/proc, CPU affinity)")
has_affinity = pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="needs os.sched_getaffinity")


# ---------------------------------------------------------------- detection
def _v2(tmp_path, rel, cpu_max=None, mem_max=None, leaf_mem=None):
    root = tmp_path / "cg"
    leaf = root / rel.strip("/")
    leaf.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("cpu memory")
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{rel}\n")
    if cpu_max:
        (leaf.parent / "cpu.max").write_text(cpu_max)      # the limit is on the parent (the job)
    if mem_max:
        (leaf.parent / "memory.max").write_text(mem_max)
    if leaf_mem:
        (leaf / "memory.max").write_text(leaf_mem)
    return root, proc


def test_scheduler_variables_in_order():
    assert sl.detect_cpus({"PBS_NCPUS": "6", "NCPUS": "4"}) == 4       # PBS Pro's NCPUS first
    assert sl.detect_cpus({"PBS_NCPUS": "6"}) == 6
    assert sl.detect_cpus({"PBS_NUM_PPN": "3"}) == 3                  # Torque
    assert sl.detect_cpus({"PBS_NP": "5"}) == 5
    assert sl.detect_cpus({"STREAM_LIMITS_MAX_THREADS": "3", "PBS_NCPUS": "6"}) == 3
    assert sl.detect_cpus({"SLURM_CPUS_PER_TASK": "5"}) == 5


def test_cgroup_v2_cpu_and_memory_use_the_lowest_on_the_path(tmp_path):
    root, proc = _v2(tmp_path, "/pbs_jobs/job1/leaf", cpu_max="250000 100000",
                     mem_max=str(8 * 10**9), leaf_mem="max")
    assert sl.detect_cpus({}, cgroup_root=root, proc_cgroup=proc) == 3        # ceil(2.5)
    assert sl.detect_memory_gb({}, cgroup_root=root, proc_cgroup=proc) == pytest.approx(8.0)


def test_cgroup_without_limits_falls_back_to_the_machine(tmp_path):
    root, proc = _v2(tmp_path, "/a/b", cpu_max="max 100000", mem_max="max")
    assert sl.detect_memory_gb({}, cgroup_root=root, proc_cgroup=proc) > 0
    assert sl._cgroup_cpu_limit(root, proc) is None
    assert sl._cgroup_memory_limit(root, proc) is None


def test_cgroup_v1(tmp_path):
    root = tmp_path / "cg"
    (root / "memory" / "job").mkdir(parents=True)
    (root / "memory" / "job" / "memory.limit_in_bytes").write_text(str(4 * 10**9))
    (root / "cpu" / "job").mkdir(parents=True)
    (root / "cpu" / "job" / "cpu.cfs_quota_us").write_text("200000")
    (root / "cpu" / "job" / "cpu.cfs_period_us").write_text("100000")
    proc = tmp_path / "proc"
    proc.write_text("4:memory:/job\n3:cpu,cpuacct:/job\n")
    assert sl.detect_memory_gb({}, cgroup_root=root, proc_cgroup=proc) == pytest.approx(4.0)
    assert sl.detect_cpus({}, cgroup_root=root, proc_cgroup=proc) == 2


def test_memory_env_overrides_everything(tmp_path):
    root, proc = _v2(tmp_path, "/j/leaf", mem_max=str(8 * 10**9))
    env = {"STREAM_LIMITS_MEMORY_GB": "12.5"}
    assert sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc) == 12.5


QSTAT = """#!/bin/sh
cat <<EOF
Job Id: 123.pbsserver
    Job_Name = fit
    Resource_List.mem = {mem}
    Resource_List.ncpus = 8
    Resource_List.vmem = {vmem}
    Resource_List.select = 1:ncpus=8:mem=16gb:
	scratch_local=50gb
EOF
"""


def _fake_qstat(tmp_path, monkeypatch, mem="16gb", vmem="32gb", script=None):
    if sys.platform == "win32":
        pytest.skip("the fake qstat is a shell script")
    exe = tmp_path / "bin" / "qstat"
    exe.parent.mkdir(exist_ok=True)
    exe.write_text(script or QSTAT.format(mem=mem, vmem=vmem))
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
    sl._qstat_resources.cache_clear()


def test_pbs_memory_from_qstat_when_nothing_else_says(tmp_path, monkeypatch):
    _fake_qstat(tmp_path, monkeypatch)
    root, proc = _v2(tmp_path, "/j/leaf")                           # no cgroup limit
    env = {"PBS_JOBID": "123.pbsserver"}
    assert sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc) == pytest.approx(
        16 * 1024**3 / 1e9)
    assert sl._memory(env, root, proc)[1] == "qstat Resource_List.mem"


def test_qstat_falls_back_to_vmem_and_reads_units(tmp_path, monkeypatch):
    _fake_qstat(tmp_path, monkeypatch, script=QSTAT.format(mem="x", vmem="512mb"))
    assert sl._qstat_memory("123.pbsserver") == (pytest.approx(512 * 1024**2 / 1e9),
                                                 "qstat Resource_List.vmem")
    assert sl._parse_pbs_size("2tb") == 2 * 1024**4
    assert sl._parse_pbs_size("1000") == 1000
    assert sl._parse_pbs_size("4mw") == 4 * 8 * 1024**2
    assert sl._parse_pbs_size("lots") is None


def test_env_and_cgroup_come_before_qstat(tmp_path, monkeypatch):
    _fake_qstat(tmp_path, monkeypatch)
    root, proc = _v2(tmp_path, "/j/leaf", mem_max=str(4 * 10**9))
    env = {"PBS_JOBID": "123.pbsserver"}
    assert sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc) == pytest.approx(4.0)
    env["STREAM_LIMITS_MEMORY_GB"] = "7"
    assert sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc) == 7.0


def test_qstat_is_not_run_without_a_job_or_when_disabled(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    _fake_qstat(tmp_path, monkeypatch, script=f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    root, proc = _v2(tmp_path, "/j/leaf")
    sl.detect_memory_gb({}, cgroup_root=root, proc_cgroup=proc)
    sl.detect_memory_gb({"PBS_JOBID": "1.x", "STREAM_LIMITS_QSTAT": "0"},
                        cgroup_root=root, proc_cgroup=proc)
    assert not marker.exists()


def test_qstat_failures_are_harmless(tmp_path, monkeypatch):
    _fake_qstat(tmp_path, monkeypatch, script="#!/bin/sh\nexit 3\n")
    assert sl._qstat_memory("9.fail") is None
    sl._qstat_resources.cache_clear()
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    assert sl._qstat_memory("9.missing") is None
    assert sl._qstat_memory("rm -rf /;") is None                    # not a job id: never run


def test_qstat_result_is_cached(tmp_path, monkeypatch):
    counter = tmp_path / "count"
    _fake_qstat(tmp_path, monkeypatch,
                script=f"#!/bin/sh\necho x >> {counter}\necho '    Resource_List.mem = 8gb'\n")
    for _ in range(3):
        assert sl._qstat_memory("5.cache")[0] == pytest.approx(8 * 1024**3 / 1e9)
    assert len(counter.read_text().split()) == 1


def test_user_named_variables_come_first_for_cpus_and_memory(tmp_path):
    root, proc = _v2(tmp_path, "/j/leaf", mem_max=str(4 * 10**9))
    env = {"STREAM_LIMITS_CPU_VARS": "SITE_CPUS, OTHER", "SITE_CPUS": "3", "NCPUS": "8",
           "STREAM_LIMITS_MEMORY_VARS": "SITE_MEM", "SITE_MEM": "2gb"}
    assert sl.detect_cpus(env, cgroup_root=root, proc_cgroup=proc) == 3
    assert sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc) == pytest.approx(
        2 * 1024**3 / 1e9)
    env["STREAM_LIMITS_MAX_THREADS"] = "2"
    assert sl.detect_cpus(env, cgroup_root=root, proc_cgroup=proc) == 2
    # an unset or unparseable named variable is skipped
    env2 = {"STREAM_LIMITS_MEMORY_VARS": "NOPE,BAD", "BAD": "lots", "STREAM_LIMITS_MEMORY_GB": ""}
    assert sl.detect_memory_gb(env2, cgroup_root=root, proc_cgroup=proc) == pytest.approx(4.0)


def test_slurm_memory_per_cpu():
    env = {"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "4"}
    assert sl.detect_memory_gb(env) == pytest.approx(2000 * 4 * 1048576 / 1e9)


def _qstat_with(tmp_path, monkeypatch, **fields):
    body = "\n".join(f"    Resource_List.{k} = {v}" for k, v in fields.items())
    _fake_qstat(tmp_path, monkeypatch, script=f"#!/bin/sh\ncat <<'EOF'\nJob Id: 7.srv\n{body}\nEOF\n")


@pytest.mark.parametrize("fields,expect,source", [
    ({"ncpus": "8", "mem": "16gb"}, 8, "qstat Resource_List.ncpus"),
    ({"select": "1:ncpus=6:mem=4gb"}, 6, "qstat Resource_List.select"),
    ({"select": "2:ncpus=4+1:ncpus=2"}, 8, "qstat Resource_List.select"),    # first chunk x count
    ({"nodes": "1:ppn=12"}, 12, "qstat Resource_List.nodes"),                # Torque
    ({"nodes": "2:ppn=3+1:ppn=1"}, 6, "qstat Resource_List.nodes"),
    ({"nodes": "3"}, 3, "qstat Resource_List.nodes"),
    ({"procs": "5"}, 5, "qstat Resource_List.procs"),
])
def test_cpus_from_qstat_when_no_variable_is_set(tmp_path, monkeypatch, fields, expect, source):
    _qstat_with(tmp_path, monkeypatch, **fields)
    root, proc = _v2(tmp_path, "/j/leaf")
    env = {"PBS_JOBID": "7.srv"}
    machine = sl._machine_cpus()
    assert sl.detect_cpus(env, cgroup_root=root, proc_cgroup=proc) == min(expect, machine)
    assert sl._cpu_limit(env, root, proc)[1] == source


def test_qstat_cpus_never_exceed_the_machine(tmp_path, monkeypatch):
    _qstat_with(tmp_path, monkeypatch, ncpus="100000")
    root, proc = _v2(tmp_path, "/j/leaf")
    assert sl.detect_cpus({"PBS_JOBID": "7.srv"}, cgroup_root=root, proc_cgroup=proc) \
        == sl._machine_cpus()


def test_cpu_variables_and_cgroups_come_before_qstat(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    _fake_qstat(tmp_path, monkeypatch, script=f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    root, proc = _v2(tmp_path, "/pbs/job/leaf", cpu_max="200000 100000")
    assert sl.detect_cpus({"PBS_JOBID": "1.x", "NCPUS": "5"}, cgroup_root=root,
                          proc_cgroup=proc) == 5
    assert sl.detect_cpus({"PBS_JOBID": "1.x"}, cgroup_root=root, proc_cgroup=proc) == 2
    assert not marker.exists()                                           # neither needed qstat
    sl.detect_cpus({"PBS_JOBID": "1.x", "STREAM_LIMITS_QSTAT": "0"},
                   cgroup_root=tmp_path / "none", proc_cgroup=tmp_path / "none")
    assert not marker.exists()                                           # and the switch is honored


def test_one_qstat_call_serves_cpus_and_memory(tmp_path, monkeypatch):
    counter = tmp_path / "count"
    _fake_qstat(tmp_path, monkeypatch, script=(
        f"#!/bin/sh\necho x >> {counter}\n"
        "echo '    Resource_List.mem = 8gb'\necho '    Resource_List.ncpus = 2'\n"))
    root, proc = _v2(tmp_path, "/j/leaf")
    env = {"PBS_JOBID": "8.srv"}
    sl.detect_cpus(env, cgroup_root=root, proc_cgroup=proc)
    sl.detect_memory_gb(env, cgroup_root=root, proc_cgroup=proc)
    sl.describe(env)
    assert len(counter.read_text().split()) == 1


def test_slurm_memory():
    assert sl.detect_memory_gb({"SLURM_MEM_PER_NODE": "16384"}) == pytest.approx(16.384 * 1.048576)


# ------------------------------------------------------------ the budget
def test_budget_headroom_is_limit_less_reserve_less_held():
    b = sl.MemoryBudget(limit_gb=10, reserve=0.2)
    assert b.usable_gb == pytest.approx(8.0)
    held = b.held_gb()
    assert b.headroom_gb() == pytest.approx(max(0.0, 8.0 - held), abs=0.2)
    assert b.headroom_gb(upcoming_gb=100) == 0.0
    assert b.fits(0.0) and not b.fits(9.0)
    assert not b.fits(1.0, share=0.0)


def test_budget_reserve_from_environment_and_validation():
    assert sl.MemoryBudget(limit_gb=8, environ={"STREAM_LIMITS_MEMORY_RESERVE": "0.4"}).reserve == 0.4
    assert sl.MemoryBudget(limit_gb=8, environ={}).reserve == sl.DEFAULT_RESERVE
    with pytest.raises(ValueError):
        sl.MemoryBudget(limit_gb=8, reserve=1.0)
    with pytest.raises(ValueError):
        sl.MemoryBudget(limit_gb=0)


def test_budget_limit_defaults_to_the_detected_memory():
    b = sl.MemoryBudget(environ={"STREAM_LIMITS_MEMORY_GB": "5"})
    assert b.limit_gb == 5.0


# --------------------------------------------------------- subprocesses
def _run(code, **env):
    e = {k: v for k, v in os.environ.items()
         if not (k.startswith(("STREAM_LIMITS", "POLARS_")) or k in (
             "PBS_NCPUS", "NCPUS", "PBS_NUM_PPN", "PBS_NP", "SLURM_CPUS_PER_TASK",
             "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"))}
    e.update(env)
    return subprocess.run([sys.executable, "-W", "always", "-c", code], env=e,
                          capture_output=True, text=True, timeout=300)


# (the last number is BLAS's threads, or "none" where threadpoolctl sees no library, as with
# the Accelerate that NumPy uses on macOS)
COUNT = ("import polars as pl, numba, pyarrow as pa, threadpoolctl as t;"
         "b = [d['num_threads'] for d in t.threadpool_info()];"
         "print(pl.thread_pool_size(), numba.get_num_threads(), pa.cpu_count(),"
         "max(b) if b else 'none')")


def _capped(out, n):
    """The probe's four numbers are all `n` (BLAS's may be absent)."""
    got = out.stdout.split()
    assert got[:3] == [str(n)] * 3 and got[3] in (str(n), "none"), (out.stdout, out.stderr)


@pytest.mark.parametrize("env", [{"STREAM_LIMITS_MAX_THREADS": "2"}, {"PBS_NCPUS": "2"}])
def test_apply_scheduler_defaults_caps_every_library(env):
    out = _run("import stream_limits as s; s.apply_scheduler_defaults();" + COUNT, **env)
    _capped(out, 2)


@has_affinity
def test_importing_changes_nothing():
    out = _run("import stream_limits;" + COUNT, PBS_NCPUS="2")
    n = len(os.sched_getaffinity(0))
    assert out.stdout.split()[0] == str(n), out.stderr


@has_affinity
def test_no_limit_changes_nothing():
    out = _run("import stream_limits as s; print(s.apply_scheduler_defaults())")
    if len(os.sched_getaffinity(0)) < os.cpu_count():
        pytest.skip("this machine restricts its CPU affinity")
    assert "threads" not in out.stdout, out.stdout


def test_a_users_own_setting_is_kept_by_the_defaults():
    out = _run("import stream_limits as s; s.apply_scheduler_defaults();" + COUNT,
               PBS_NCPUS="4", POLARS_MAX_THREADS="3")
    assert out.stdout.split()[:2] == ["3", "4"], out.stderr


def test_explicit_call_before_polars_limits_polars():
    out = _run("import stream_limits; stream_limits.limit_threads(3);" + COUNT)
    assert out.stdout.split()[0] == "3", out.stderr
    assert "Polars was imported" not in out.stderr


def test_polars_already_imported_warns():
    out = _run("import polars; polars.thread_pool_size(); import stream_limits; stream_limits.limit_threads(1)")
    assert "Polars was imported before the thread limit" in out.stderr


def test_thread_scope_restores():
    import numba
    import numpy as np  # noqa: F401  (loads BLAS)
    import threadpoolctl
    before_nb = numba.get_num_threads()
    blas = lambda: [d["num_threads"] for d in threadpoolctl.threadpool_info()   # noqa: E731
                    if d["user_api"] == "blas"]
    before_blas = blas()
    with sl.thread_scope(1):
        assert numba.get_num_threads() == 1 and all(n == 1 for n in blas())
    assert numba.get_num_threads() == before_nb and blas() == before_blas
    with sl.thread_scope(None):
        assert numba.get_num_threads() == before_nb


def test_thread_scope_restores_after_an_error():
    import numba
    before = numba.get_num_threads()
    with pytest.raises(RuntimeError):
        with sl.thread_scope(1):
            raise RuntimeError
    assert numba.get_num_threads() == before


# ----------------------------------------------------------- polars budget
needs_polars2 = pytest.mark.skipif(not sl.polars_spills(), reason="needs Polars 2.0 or later")


@needs_polars2
def test_limit_memory_sets_a_budget_before_polars():
    out = _run("import os, stream_limits as s; r = s.limit_memory(10); "
               "print(r['budget_mb'], os.environ['POLARS_OOC_MEMORY_BUDGET_MB'])")
    assert out.stdout.split() == ["4250", "4250"], out.stderr           # 0.5 x (10 - 1.5) GB


@needs_polars2
def test_limit_memory_floor_and_spill_dir(tmp_path):
    out = _run(f"import os, stream_limits as s; r = s.limit_memory(1, spill_dir={str(tmp_path)!r}); "
               "print(r['budget_mb'], os.environ['POLARS_OOC_SPILL_DIR'])")
    assert out.stdout.split() == ["250", str(tmp_path)], out.stderr


@needs_polars2
def test_limit_memory_respects_the_users_budget_and_the_off_switch():
    a = _run("import stream_limits as s; print(s.limit_memory(10)['skipped'])",
             POLARS_OOC_MEMORY_BUDGET_MB="123")
    assert "already set" in a.stdout, a.stderr
    b = _run("import stream_limits as s; print(s.limit_memory(10)['skipped'])",
             STREAM_LIMITS_POLARS_SPILL="0")
    assert "STREAM_LIMITS_POLARS_SPILL" in b.stdout, b.stderr


@needs_polars2
def test_limit_memory_after_polars_warns():
    out = _run("import polars; import stream_limits as s; r = s.limit_memory(10); print(r['skipped'])")
    assert "Polars was imported first" in out.stdout and "out-of-core budget" in out.stderr


@needs_polars2
def test_apply_defaults_sets_the_budget_only_with_a_known_limit():
    a = _run("import os, stream_limits as s; s.apply_scheduler_defaults();"
             "print(os.environ.get('POLARS_OOC_MEMORY_BUDGET_MB'))", STREAM_LIMITS_MEMORY_GB="8")
    assert a.stdout.strip() == "3250", a.stderr
    b = _run("import os, stream_limits as s; s.apply_scheduler_defaults();"
             "print(os.environ.get('POLARS_OOC_MEMORY_BUDGET_MB'))")
    if sl.describe()["memory_source"] in ("physical memory", "default"):
        assert b.stdout.strip() == "None", b.stderr


# A streaming sort of a Parquet file, in a fresh process so the peak is its own. The file is
# large enough that the sort, not the readers' fixed overhead, sets the peak: at 8 million
# rows the budget's effect was borderline on a 4-core runner (a ratio of 0.75 to 0.81),
# at 20 million it is 0.4 to 0.5.
SORT = r'''
import sys
import stream_limits as s
path, mode = sys.argv[1], sys.argv[2]
if mode == "budget":
    s.limit_memory(2.0, fraction=0.1)
import polars as pl
s.reset_peak_rss()
pl.scan_parquet(path).sort("a").sink_parquet(path + ".sorted", engine="streaming")
print(s.peak_rss_gb())
'''


# (generated in a subprocess, so that polars is never imported into the pytest process: with it
# loaded, every in-process limit_threads() call warns that Polars can no longer be capped)
GENERATE = r'''
import sys
import numpy as np, polars as pl
n = 20_000_000
rng = np.random.default_rng(0)
pl.DataFrame({"a": rng.integers(0, 1_000_000, n), **{f"x{i}": rng.normal(size=n) for i in range(6)}}
             ).write_parquet(sys.argv[1], row_group_size=500_000)
'''


@linux_only      # the peak-memory counter can be reset only on Linux
@needs_polars2
def test_the_budget_bounds_a_streaming_sort(tmp_path):
    path = str(tmp_path / "in.parquet")
    made = subprocess.run([sys.executable, "-c", GENERATE, path], capture_output=True, text=True,
                          timeout=600)
    assert made.returncode == 0, made.stderr[-400:]
    peaks = {}
    for mode in ("none", "budget"):
        out = subprocess.run([sys.executable, "-c", SORT, path, mode], capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-400:]
        peaks[mode] = float(out.stdout.split()[-1])
    assert peaks["budget"] < 0.8 * peaks["none"], peaks


# --------------------------------------------------------------- set_limits
def test_set_limits_with_values_applies_and_records_them():
    out = _run("import os, stream_limits as s; r = s.set_limits(threads=3, memory_gb=12);"
               "print(sorted(r), os.environ['STREAM_LIMITS_MAX_THREADS'],"
               "os.environ['STREAM_LIMITS_MEMORY_GB'], s.detect_cpus(), s.detect_memory_gb());"
               + COUNT)
    first, counts = out.stdout.strip().splitlines()[-2:]
    assert first == "['memory', 'threads'] 3 12.0 3 12.0", (out.stdout, out.stderr)
    assert counts.split()[:3] == ["3", "3", "3"], out.stderr


@needs_polars2
def test_set_limits_sets_polars_budget_from_the_given_memory():
    out = _run("import os, stream_limits as s; s.set_limits(memory_gb=10);"
               "print(os.environ['POLARS_OOC_MEMORY_BUDGET_MB'])")
    assert out.stdout.strip() == "4250", out.stderr


def test_set_limits_without_values_uses_the_jobs_allocation():
    out = _run("import stream_limits as s; r = s.set_limits(); print(sorted(r));" + COUNT,
               PBS_NCPUS="2", STREAM_LIMITS_MEMORY_GB="9")
    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "['memory', 'threads']", (out.stdout, out.stderr)
    assert lines[-1].split()[:3] == ["2", "2", "2"], out.stderr


@has_affinity
def test_set_limits_without_values_or_a_limit_does_nothing():
    if len(os.sched_getaffinity(0)) < os.cpu_count() or sl.describe()["memory_source"] not in (
            "physical memory", "default"):
        pytest.skip("this machine has a restricted CPU affinity or a memory limit")
    out = _run("import os, stream_limits as s; print(s.set_limits(),"
               "os.environ.get('POLARS_MAX_THREADS'), os.environ.get('POLARS_OOC_MEMORY_BUDGET_MB'))")
    assert out.stdout.strip() == "{} None None", out.stderr


def test_set_limits_is_what_the_libraries_call_with_no_arguments():
    a = _run("import stream_limits as s; print(s.set_limits())", PBS_NCPUS="2")
    b = _run("import stream_limits as s; print(s.apply_scheduler_defaults())", PBS_NCPUS="2")
    assert a.stdout == b.stdout


def test_set_limits_warning_points_at_the_caller():
    out = _run("import polars\npolars.thread_pool_size()\nimport stream_limits\nstream_limits.set_limits(threads=1)")
    assert "<string>:4" in out.stderr and "Polars was imported before" in out.stderr, out.stderr


# ------------------------------------------------------------- peak memory
@linux_only
def test_peak_rss_tracks_a_big_allocation():
    import numpy as np
    sl.reset_peak_rss()
    base = sl.peak_rss_gb()
    a = np.ones(50_000_000)          # 400 MB, touched
    peak = sl.peak_rss_gb()
    del a
    assert peak is not None and peak - base > 0.3


def test_describe_and_module_entry_point():
    d = sl.describe()
    assert d["cpus"] >= 1 and d["memory_gb"] > 0
    out = subprocess.run([sys.executable, "-m", "stream_limits"], capture_output=True, text=True)
    assert '"cpus"' in out.stdout


# ------------------------------------------------------------ Windows memory counters
def test_windows_counters_are_used_when_present(monkeypatch):
    # the branch, on any platform: working set 1 GB, peak working set 2 GB
    monkeypatch.setattr(sl, "_windows_memory", lambda: (10**9, 2 * 10**9))
    assert sl.peak_rss_gb() == pytest.approx(2.0)
    assert sl.current_rss_gb() == pytest.approx(1.0)


def test_off_windows_there_is_no_windows_reading():
    if sys.platform != "win32":
        assert sl._windows_memory() is None


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows API")
def test_windows_reports_current_and_peak_memory():
    # a fresh process, so the lifetime peak is its own
    code = ("import numpy as np, stream_limits as s\n"
            "base = s.peak_rss_gb(); now = s.current_rss_gb()\n"
            "a = np.ones(50_000_000)\n"                         # 400 MB, touched
            "peak = s.peak_rss_gb(); held = s.current_rss_gb()\n"
            "del a\n"
            "print(base, now, peak, held, s.peak_rss_gb())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    base, now, peak, held, after = map(float, out.stdout.split())
    assert 0 < now <= base                      # current never exceeds the peak
    assert peak - base > 0.3 and held - now > 0.3
    assert after >= peak - 1e-6                 # the peak does not fall when memory is freed
