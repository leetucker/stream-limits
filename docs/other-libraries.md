# Other libraries

`stream-limits` covers a library when the limit can be set **for the whole process**, either by an
environment variable read at import or by a call that changes a global setting. Many popular
libraries have neither: their limits belong to an object (a database connection, a cluster) that
you create. For those, this page has recipes, using the numbers `stream-limits` finds.

```python
import stream_limits as sl

n = sl.detect_cpus()               # the job's CPU allocation
total = sl.detect_memory_gb()      # the job's memory limit
usable = sl.MemoryBudget().usable_gb   # the limit less the reserve (default 15%)
```

```{admonition} What has been checked
:class: note

**Verified against the library, in the test suite:** Polars, numba, BLAS (through OpenBLAS and
threadpoolctl), pyarrow, numexpr.

**Set by environment variable only, with the names taken from each library's documentation and not
checked against the library here:** BLIS, OpenMP's `OMP_THREAD_LIMIT`, rayon, TensorFlow, joblib's
loky, Julia. An environment variable for a library that is not installed does nothing, so setting it
is harmless; whether the library obeys it is that library's business.

**Lowered through its API, checked only against a stand-in for the API:** PyTorch's
`set_num_threads`.

**The recipes below are not run by the test suite**, because the libraries are not dependencies.
Check them against the version you use.
```

## What is covered, and how

| Library | Covered by | Notes |
|---|---|---|
| Polars | `POLARS_MAX_THREADS`; `POLARS_OOC_MEMORY_BUDGET_MB` for memory | fixed at import; see [Thread limits](threads.md) and [Memory limits](memory.md) |
| numba | `NUMBA_NUM_THREADS`; `set_num_threads` | lowered, never raised |
| NumPy, SciPy (OpenBLAS, MKL, BLIS, Accelerate) | `OPENBLAS_`, `MKL_`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`; `threadpoolctl` | |
| OpenMP programs (LightGBM, XGBoost, scikit-learn's compiled code) | `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT` | their own `n_jobs`/`nthread` arguments, when given, win |
| pyarrow | `set_cpu_count`, `set_io_thread_count` | |
| numexpr (and pandas `eval`) | `NUMEXPR_NUM_THREADS`, `NUMEXPR_MAX_THREADS`; `set_num_threads` | |
| PyTorch (intra-op) | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`; `torch.set_num_threads` | inter-op threads and `DataLoader` workers: below |
| TensorFlow | `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` | read when TensorFlow initializes |
| Rust libraries on rayon | `RAYON_NUM_THREADS` | not Polars, which has its own pool |
| joblib, and so scikit-learn's `n_jobs=-1` | `LOKY_MAX_CPU_COUNT` | the worker count it counts CPUs from |
| Julia, started from Python | `JULIA_NUM_THREADS` | a child process inherits the environment |

Variables you have already set are left alone, and loaded libraries are only lowered, unless you
pass `override=True`; see [Thread limits](threads.md#variables-you-have-set).

## DuckDB

The thread count and the memory limit are settings of a connection. Pass them when you connect:

```python
import duckdb
import stream_limits as sl

con = duckdb.connect(config={
    "threads": sl.detect_cpus(),
    "memory_limit": f"{int(sl.MemoryBudget().usable_gb)}GB",
    "temp_directory": "/path/to/job/scratch",     # where it spills past the limit
})
```

or on an existing connection, `con.execute("SET threads = 8")` and
`con.execute("SET memory_limit = '12GB'")`. DuckDB spills larger-than-memory work to
`temp_directory`; put that on the job's scratch space.

## Dask

The local threaded scheduler takes a worker count; a `LocalCluster` takes workers, threads per
worker and a memory limit **per worker**:

```python
import dask
import stream_limits as sl

dask.config.set(num_workers=sl.detect_cpus())          # threaded scheduler

from dask.distributed import LocalCluster
cluster = LocalCluster(
    n_workers=1, threads_per_worker=sl.detect_cpus(),
    memory_limit=f"{sl.MemoryBudget().usable_gb:.0f}GB",   # per worker: n_workers x this
)
```

## Ray

Ray detects the node's resources, not the job's, unless you tell it:

```python
import ray
import stream_limits as sl

ray.init(num_cpus=sl.detect_cpus(),
         object_store_memory=int(0.3 * sl.MemoryBudget().usable_gb * 1e9))
```

## JAX

JAX's CPU backend has no process-wide thread setting of its own. A flag commonly used to bound
its Eigen thread pool is set before JAX is imported; it has changed between versions, so check it
against yours:

```python
import os
import stream_limits as sl

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    f"intra_op_parallelism_threads={sl.detect_cpus()}"
)
import jax
```

## joblib, multiprocessing, concurrent.futures

These size their pools from the CPU count unless you pass one. `os.cpu_count()` is the machine's
(it ignores a scheduler's allocation and the CPU affinity; `os.process_cpu_count()`, Python 3.13 and
later, honors the affinity but not a scheduler's allocation), so pass the allocation:

```python
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel
import stream_limits as sl

with ProcessPoolExecutor(max_workers=sl.detect_cpus()) as pool: ...
Parallel(n_jobs=sl.detect_cpus())(...)
```

`stream_limits.set_limits()` sets `LOKY_MAX_CPU_COUNT`, which covers joblib's `n_jobs=-1` and
scikit-learn's.

## PyTorch beyond intra-op threads

`set_limits` lowers `torch.set_num_threads` (intra-op). Inter-op threads can be set only before any
parallel work has run, and `DataLoader` workers are an argument:

```python
import torch
import stream_limits as sl

torch.set_num_interop_threads(2)                  # before the first parallel op
loader = torch.utils.data.DataLoader(data, num_workers=max(1, sl.detect_cpus() - 1))
```

## Spark and the JVM

Local mode takes a core count and the driver's heap size is fixed when the JVM starts:

```python
from pyspark.sql import SparkSession
import stream_limits as sl

spark = (SparkSession.builder
         .master(f"local[{sl.detect_cpus()}]")
         .config("spark.driver.memory", f"{int(sl.MemoryBudget().usable_gb)}g")
         .getOrCreate())
```

## A general backstop: CPU affinity (Linux)

For a library with no setting at all, or to make sure a cap holds whatever a library does, restrict
which cores the process may run on. Set before the libraries are imported, it makes **Polars, numba,
pyarrow and OpenBLAS size their pools to the cores you left** (checked here: started on 2 of 8 cores,
each reported 2 threads), and the operating system enforces the rest:

```python
import os
import stream_limits as sl

cores = sorted(os.sched_getaffinity(0))[: sl.detect_cpus()]
os.sched_setaffinity(0, cores)          # before importing polars, numba, ...
```

or start the program with `taskset -c 0-7 python script.py`. Two limits: it constrains *where*
threads run, not how many exist, and a library that does not read the affinity (numexpr, checked
here, still used all 8) keeps its full pool of threads, queueing on the cores you left. It backs the
settings above up; it does not replace them. Child processes inherit it. It is not available on
macOS or Windows.

## What not to do: `RLIMIT_AS`

`resource.setrlimit(resource.RLIMIT_AS, ...)` would give a hard memory cap that raises
`MemoryError` instead of being killed. It counts *virtual address space*, which memory-mapped files,
numba's JIT and allocators such as Polars' reserve far beyond what they use, so it fails jobs that
fit comfortably in memory. It is not recommended, and `stream-limits` does not use it.
