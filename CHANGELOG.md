# Changelog

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
