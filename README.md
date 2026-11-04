# stream-limits

Cap the threads and memory of numeric Python libraries at a job's allocation, with one
interface, and measure the memory a block of code used.

On a cluster (PBS, Slurm, a container) a job gets a fixed number of CPUs and a fixed
amount of memory, and is killed if it exceeds either. Polars, numba, BLAS and pyarrow
each size their thread pools separately and default to every core of the *node*; code
that asks how much memory there is sees the node's too. `stream-limits` finds the job's
allocation, applies it to each library as far as the library allows, and says so when it
cannot. It depends on nothing outside the standard library and imports none of the
libraries it limits, so it can run before they are imported, which is when some of them
(Polars) need to be told.

It is what [hdfe-stream](https://github.com/leetucker/hdfe-stream),
[logit-stream](https://github.com/leetucker/logit-stream) and
[dpd-stream](https://github.com/leetucker/dpd-stream) use, and it works on its own.
Documentation: <https://leetucker.github.io/stream-limits/> (build locally with
`pip install -e ".[docs]"` and `sphinx-build -W -b html docs docs/_build/html`).

```python
import stream_limits
stream_limits.set_limits()         # the job's allocation; or set_limits(threads=8, memory_gb=16)

import polars as pl                # now capped
import numpy as np
```

```
$ python -m stream_limits          # what was found, from inside a job
{"cpus": 8, "cpus_source": "PBS_NCPUS", "memory_gb": 32.0, "memory_source": "cgroup memory limit", ...}
```

Variables you have set are **not overridden** by default (a warning names any you set above the cap);
loaded libraries are only lowered. `override=True` replaces them. More libraries are covered than the
table below (numexpr, PyTorch, OpenMP programs, TensorFlow, rayon, joblib, Julia), and the documentation's
"Other libraries" has recipes for those with only per-object settings (DuckDB, Dask, Ray, JAX, Spark).

## What each library allows

| Library | Can it be capped after import? | What `limit_threads` does |
|---|---|---|
| Polars | **No**: the pool is sized once, at import (Polars 1.37: at first use) | sets `POLARS_MAX_THREADS` (before import); warns if too late |
| numba | down, never above `NUMBA_NUM_THREADS` at import | sets the variable; lowers `set_num_threads` |
| BLAS (NumPy, SciPy) | yes, through `threadpoolctl` | sets `OMP_`/`OPENBLAS_`/`MKL_NUM_THREADS`; `threadpool_limits` |
| pyarrow | yes | `set_cpu_count`, `set_io_thread_count` |

`thread_scope(n)` holds numba and BLAS to `n` threads for a block and restores them.

## Memory

- `detect_memory_gb()` is the memory the process may use: `STREAM_LIMITS_MEMORY_GB`, else the
  cgroup limit (the lowest on the path to the root, so a scheduler's job limit is found), else
  PBS's own record of the job, else `SLURM_MEM_PER_NODE`, else the machine's. PBS sites often set
  no memory variable and enforce no cgroup; there `qstat -f $PBS_JOBID` is run once (no shell, a
  10 s timeout, cached) and `Resource_List.mem` (else `vmem`) is read, with PBS's size suffixes.
  `STREAM_LIMITS_QSTAT=0` turns this off.
- `limit_memory()` gives Polars 2.x an out-of-core budget (`POLARS_OOC_MEMORY_BUDGET_MB`),
  past which its streaming engine spills to disk instead of growing. A streaming sort of 1.9 GB
  of Parquet peaked at 6.2 GB with the defaults and 1.8 GB with a 400 MB budget, in twice the
  time. The budget bounds what Polars' operators hold, not the whole process (its Parquet
  readers hold about 1.5 GB more on 8 threads). This is undocumented in Polars 2.x and read at
  import, so it is set only for Polars 2.0 or later, only before Polars is imported, never over
  a budget you set, and `STREAM_LIMITS_POLARS_SPILL=0` turns it off.
- `MemoryBudget(limit_gb, reserve)` is for code that chooses between a fast path that needs
  memory and a slower one that needs less: `budget.headroom_gb()` is the limit, less a reserve
  (default 15%, `STREAM_LIMITS_MEMORY_RESERVE`) for what the code does not model, less what the
  process holds now. Ask whether your exact allocation fits, not whether it is under a fixed
  fraction of the machine.
- `peak_rss_gb()` / `reset_peak_rss()` read the system's own high-water mark (Linux, macOS and
  Windows): free, no polling.

`set_limits` is `limit_threads` and `limit_memory` in one call: values you give are applied, values you
leave out are the job's allocation if one is found and nothing otherwise. The two remain, for their
own options (`spill_dir`, `fraction`, ...).

## Automatically

`apply_scheduler_defaults()` (which is `set_limits()` with no arguments) does `limit_threads` and `limit_memory`, only for limits it finds
(a variable, a scheduler's environment, a cgroup, a restricted CPU affinity), and sets only
variables you have not set. Libraries built on this call it when they are imported; importing
`stream_limits` itself changes nothing.

## Environment variables

| Variable | Meaning |
|---|---|
| `STREAM_LIMITS_MAX_THREADS` | the thread cap |
| `STREAM_LIMITS_MEMORY_GB` | the memory limit |
| `STREAM_LIMITS_MEMORY_RESERVE` | `MemoryBudget`'s reserve (default 0.15) |
| `STREAM_LIMITS_POLARS_SPILL` | `0`: do not set Polars' budget |
| `STREAM_LIMITS_POLARS_FRACTION` | the budget's share of the limit above the readers' floor (default 0.5) |
| `STREAM_LIMITS_QSTAT` | `0`: do not run `qstat` to find a PBS job's CPUs and memory |
| `STREAM_LIMITS_CPU_VARS` | names of variables holding a CPU count, checked first (comma-separated) |
| `STREAM_LIMITS_MEMORY_VARS` | the same for memory; each holds a size (`16gb`) |
| `STREAM_LIMITS_SPILL_DIR` | where Polars spills (default its temp dir: on a cluster, check `$TMPDIR`) |

Thread limits are read from, in order: the argument; `STREAM_LIMITS_MAX_THREADS`; the variables named
in `STREAM_LIMITS_CPU_VARS`; `NCPUS`, `PBS_NCPUS`, `PBS_NUM_PPN`, `PBS_NP`; `SLURM_CPUS_PER_TASK`,
`SLURM_CPUS_ON_NODE`; the cgroup CPU quota; `qstat -f $PBS_JOBID`; the CPU affinity; `os.cpu_count()`.
See the documentation's "Schedulers and containers" for each source and for sites that do it differently.

## Install

```
pip install stream-limits            # nothing else is installed
pip install "stream-limits[blas]"    # threadpoolctl, to limit BLAS after NumPy is loaded
```

Linux and macOS are tested on Python 3.10 to 3.15, Windows on 3.10 and 3.14. On Windows the module imports,
applies thread limits and reads memory use (peak and current working set), but it detects no cgroup;
macOS and Windows report a lifetime peak, as their counters cannot be reset. On macOS, NumPy's Apple
Accelerate BLAS is limited only by `VECLIB_MAXIMUM_THREADS` before NumPy is imported (threadpoolctl cannot see it).

## Running a program under a memory limit

The optional launcher (`pip install 'stream-limits[launcher]'`, which adds psutil)
runs a program in a child process, stops it if it goes over a memory or time
limit, and reports why it ended and its peak memory:

```bash
python -m stream_limits.launcher --mode watchdog --memory-gb 4 --threads 2 -- python analysis.py
```

The watchdog works on Linux, macOS and Windows; a kernel-enforced limit needs a
Linux cgroup set up for it. `stream-limits[testing]` adds pytest fixtures for
memory tests. See [Running a program under a memory limit](docs/launcher.md).

MIT licence.
