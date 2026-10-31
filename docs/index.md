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

It is what [hdfe-stream](https://leetucker.github.io/hdfe-stream/),
[logit-stream](https://leetucker.github.io/logit-stream/) and
[dpd-stream](https://leetucker.github.io/dpd-stream/) use, and it works on its own.

```python
import stream_limits
stream_limits.set_limits()         # the job's allocation; or set_limits(threads=8, memory_gb=16)

import polars as pl                # now capped
import numpy as np
```

`set_limits` is {py:func}`~stream_limits.limit_threads` and {py:func}`~stream_limits.limit_memory`
in one call. Values you give are applied; values you leave out are the job's allocation if one is
found, and nothing otherwise, so the same line is right on a cluster and on a laptop. The two
functions it calls remain, for the options the one call leaves out.

```
$ python -m stream_limits          # what was found, from inside a job
{
  "cpus": 8,
  "cpus_source": "NCPUS",
  "memory_gb": 17.18,
  "memory_source": "qstat Resource_List.mem",
  ...
}
```

## Where to start

- **On a cluster, before anything else**: run `python -m stream_limits` inside a job and check
  that the CPUs and memory it reports are the ones you asked for. If they are not, see
  [Schedulers and containers](schedulers.md), which says where each number comes from and how
  to tell the library about a site that does it differently.
- **To cap threads**: [Thread limits](threads.md).
- **For a library that is not covered** (DuckDB, Dask, Ray, JAX, Spark, ...): [Other libraries](other-libraries.md).
- **To bound memory**: [Memory limits](memory.md), which covers the memory a job has,
  Polars' spill-to-disk budget, and `MemoryBudget`, for code that chooses between a fast path
  that needs memory and a slower one that needs less.
- **If you use hdfe-stream, logit-stream or dpd-stream**: they call
  `apply_scheduler_defaults()` when imported, so a job is capped without a line of your code,
  and their `max_memory_gb` and `n_threads` options use what is described here. Their
  `profile=True` option, which estimates a fit's memory before you run it, is documented in
  [hdfe-stream](https://leetucker.github.io/hdfe-stream/resources.html).

## Automatically, or explicitly

`apply_scheduler_defaults()` (which is `set_limits()` with no arguments) does `limit_threads` and
`limit_memory`, **only for limits it finds** (a variable, a scheduler's environment, a cgroup, a restricted CPU affinity), and sets
only variables you have not set yourself; with no limit it does nothing, so a workstation is
untouched. Libraries built on `stream-limits` call it when they are imported. Importing
`stream_limits` itself changes nothing.

## Install

```
pip install stream-limits            # nothing else is installed
pip install "stream-limits[blas]"    # threadpoolctl, to limit BLAS after NumPy is loaded
```

```{toctree}
:hidden:

threads
memory
schedulers
other-libraries
reference
changelog
```
