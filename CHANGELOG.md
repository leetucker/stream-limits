# Changelog

## 0.2.1

- **The watchdog no longer counts the pages of memory-mapped files.** A program
  that memory-maps large scratch files, as hdfe-stream does, has those pages in
  its resident memory, but the kernel can reclaim them, so a watchdog that
  counted them stopped healthy programs: a 20-million-row hdfe-stream fit with
  1.2 GB of its own memory reached 2.3 GB of RSS. The watchdog now limits
  private memory (Linux, Windows; macOS still uses RSS), reported as
  `peak_private_bytes`. `--memory-measure rss` restores the old behaviour.
- On Windows, `os_peak_rss_bytes` is now the program's exact peak working set,
  read when it exits, instead of the last value seen while sampling.

## 0.2.0

- **Run a program under a memory limit and see how much memory it used.** The new
  launcher runs a script, test suite or other program in a child process, stops it
  if it goes over a memory limit or a time limit, and reports why it ended and its
  peak memory. The limit is a watchdog that checks memory every 10 ms (Linux,
  macOS, Windows), or a kernel limit through a Linux cgroup. It also sets the
  program's threads (`--threads`). It is an optional extra,
  `pip install 'stream-limits[launcher]'`, which adds psutil; the rest of
  stream-limits is unchanged and still needs only the standard library. See
  [Running a program under a memory limit](https://leetucker.github.io/stream-limits/launcher.html).
- **Pytest fixtures for memory tests.** `pytest_plugins =
  ["stream_limits.pytest_plugin"]` gives a test suite `resource_runner`, which runs
  a test's work in a child process under the watchdog on every platform, or under a
  kernel limit when one is set up (`STREAM_LIMITS_TEST_CGROUP_PARENT`). Install
  `stream-limits[testing]`.

## 0.1.1

- **Windows reports memory use.** `peak_rss_gb` and `current_rss_gb` returned `None` on Windows in
  0.1.0; they now read the process's peak and current working set (the physical memory resident in it,
  as on Linux) through the Windows API, with no new dependency. Windows cannot reset its peak, so
  there it is the process's lifetime peak, as on macOS. `MemoryBudget` uses the real current memory
  on Windows instead of assuming 0.5 GB.
- Documentation: Polars 1.37 creates its thread pool at first use, while later versions (1.44 and
  2.x) create it at import; the thread-limit pages say so.

## 0.1.0

First release, coinciding with updates to hdfe-stream, logit-stream and
dpd-stream to use it. Anyone running other numeric code on a scheduler can use
it on its own.

**Limits**

- `set_limits(threads=None, memory_gb=None)` applies thread and memory limits in one call. Values you
  give are used; values you omit are the job's allocation if one is found, and nothing otherwise.
- `limit_threads` caps Polars, numba, NumPy/SciPy's BLAS (OpenBLAS, MKL, BLIS, Accelerate), OpenMP
  programs (LightGBM, XGBoost, scikit-learn), pyarrow, numexpr, PyTorch, TensorFlow, rayon-based Rust
  libraries, joblib and Julia. It warns when Polars was imported first and can no longer be capped.
- `limit_memory` gives Polars 2.x an out-of-core memory budget, so its streaming engine spills to disk
  instead of growing (a streaming sort of 1.9 GB peaked at 6.2 GB; with a budget, 1.8 GB).
- `apply_scheduler_defaults` does both, only when a limit is found. The packages built on this one call
  it when imported.
- Variables you have set are left alone, and loaded libraries are only lowered; a variable set above the
  cap is named in a warning. `override=True` replaces them.
- `thread_scope` holds numba and BLAS to a thread count for one block of code.
- Every setting can be given as a `STREAM_LIMITS_*` environment variable (`STREAM_LIMITS_MAX_THREADS`,
  `STREAM_LIMITS_MEMORY_GB`, ...), which is how a job script sets it. Values passed to `set_limits` are
  recorded there too, so the rest of the process, child processes and the packages built on this one agree.

**Finding the allocation**

- `detect_cpus` and `detect_memory_gb` read the environment, cgroup v1/v2 limits (the lowest on the path
  to the root), PBS (`NCPUS`, `PBS_NUM_PPN`, ...; `qstat -f $PBS_JOBID` when no variable or cgroup says)
  and Slurm.
- `STREAM_LIMITS_CPU_VARS` and `STREAM_LIMITS_MEMORY_VARS` name a site's own variables.
- `describe` and `python -m stream_limits` show what was found and where it came from.

**Memory**

- `MemoryBudget` gives the headroom left, for code that chooses between a fast path that needs memory and
  a slower one that needs less.
- `peak_rss_gb`, `reset_peak_rss` and `current_rss_gb` read the kernel's own counters.

**Documentation**

- A Sphinx site, built in CI: thread limits, memory limits, schedulers and containers, and recipes for
  libraries with only per-object settings (DuckDB, Dask, Ray, JAX, Spark).
