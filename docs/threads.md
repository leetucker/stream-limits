# Thread limits

Polars, numba, BLAS and pyarrow each keep their own thread pool, and each defaults to every
core of the machine. A job allocated 8 of a node's 64 cores that imports all four can run 64
threads and be killed for it.

## What each library allows

| Library | Can it be capped after import? | What `limit_threads` does |
|---|---|---|
| Polars | **No**: the pool is sized once, at import (Polars 1.37: at first use) | sets `POLARS_MAX_THREADS` (before import); warns if too late |
| numba | down, never above `NUMBA_NUM_THREADS` at import | sets the variable; lowers `set_num_threads` |
| BLAS (NumPy, SciPy) | yes, through `threadpoolctl` | sets `OMP_`/`OPENBLAS_`/`MKL_NUM_THREADS`; `threadpool_limits` |
| Apple Accelerate (NumPy on macOS) | **No**: `threadpoolctl` cannot see it | sets `VECLIB_MAXIMUM_THREADS` (before NumPy is imported) |
| pyarrow | yes | `set_cpu_count`, `set_io_thread_count` |
| numexpr, PyTorch | yes | `NUMEXPR_NUM_THREADS`/`_MAX_THREADS` (before import); `set_num_threads` |
| OpenMP programs, TensorFlow, rayon, joblib, Julia | by environment variable, before they start | `OMP_THREAD_LIMIT`, `TF_NUM_*_THREADS`, `RAYON_NUM_THREADS`, `LOKY_MAX_CPU_COUNT`, `JULIA_NUM_THREADS` |

[Other libraries](other-libraries.md) lists everything covered, what was checked against the library
and what was not, and has recipes for libraries whose limits belong to an object (DuckDB, Dask, Ray, ...).

So the only reliable way to cap everything is to set environment variables **before** the
libraries are imported. {py:func}`stream_limits.limit_threads` does that, and then lowers what
can still be lowered in libraries that are already loaded. ({py:func}`stream_limits.set_limits`
does this and the memory limit in one call.)

```python
import stream_limits
stream_limits.limit_threads(8)        # or limit_threads(): the job's allocation

import polars as pl                   # now capped at 8
```

If Polars was imported first, `limit_threads` still lowers numba, BLAS and pyarrow, cannot
touch Polars, and says so in a `RuntimeWarning` that names the fix. The fix is the environment
variable, set in the job script where it is in place before any Python starts:

```bash
export STREAM_LIMITS_MAX_THREADS=$NCPUS
```

The call returns what it did (`set_env`, `kept`, `lowered`, `too_late`), for a log line.

(variables-you-have-set)=
## Variables you have set

By default `stream-limits` **does not override a variable you have set**. If `OMP_NUM_THREADS=1`
is in your environment (to keep BLAS single-threaded, say), `limit_threads(8)` leaves it at 1, lists
it under `kept` in the report, and sets the others. Libraries that are already loaded are only ever
*lowered*: a cap is a ceiling, so one you set lower yourself stays.

The risk is the opposite one, a stale variable above the cap (an `OMP_NUM_THREADS=64` in a
`.bashrc`). So a variable you set above the cap is left alone **and a warning names it**:

```
RuntimeWarning: OMP_NUM_THREADS=64 is set above the thread limit 8 and left alone; pass
override=True (or unset it) to apply the limit.
```

`override=True` replaces the variables and sets loaded libraries to exactly the cap (numba cannot go
above its import-time maximum). The same holds for the memory side: a Polars budget or spill
directory you have set is kept unless `override=True`. `STREAM_LIMITS_MAX_THREADS` and
`STREAM_LIMITS_MEMORY_GB` are the exception: they are this library's own, and an argument you give
always wins over them.

## For one block of code

{py:func}`stream_limits.thread_scope` holds numba and BLAS to `n` threads inside a block and
restores them after, also when the block raises. Polars is not affected: its pool is fixed (at import, or at first use in Polars 1.37) for
the process.

```python
with stream_limits.thread_scope(2):
    result = heavy_numba_function(data)
```

## How many threads?

With no argument, `limit_threads` uses {py:func}`stream_limits.detect_cpus`: the job's CPU
allocation, read from the environment, cgroups or the scheduler (see
[Schedulers and containers](schedulers.md)), and otherwise the machine's. A job that runs several
processes should give each its share, for example
`STREAM_LIMITS_MAX_THREADS=$(( NCPUS / 4 ))` for four.


## How OpenMP threads wait

OpenMP threads that finish a parallel region either spin, waiting for the next one, or
sleep. Spinning is quicker to resume, but a program that makes many short parallel calls
(numba's `prange` loops, one per batch of data) can run several times slower with it,
notably on CPUs that mix performance and efficiency cores, and a little slower on others.
Programs with few, long parallel regions are unaffected: hdfe-stream's solvers and
logit-stream's fits ran as fast either way.

{py:func}`stream_limits.set_openmp_wait_policy` sets `OMP_WAIT_POLICY` (to `"passive"` unless you
say `"active"`). A package opts in by calling it, because the setting reaches every
OpenMP library in the process (BLAS builds on OpenMP, scikit-learn, LightGBM); importing
stream-limits, and `apply_scheduler_defaults`, leave it alone. A value you have set is kept unless
you pass `override=True`. The runtime reads the variable when it loads, which for numba is
at its first parallel call, so call the function before that. The result says when a runtime
was already loaded (Linux), in which case the variable is set for child processes only.
On Windows, numba uses Microsoft's OpenMP runtime, which does not read the variable, so the
call changes nothing there.
