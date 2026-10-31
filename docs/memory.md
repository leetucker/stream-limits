# Memory limits

## The memory a job has

{py:func}`stream_limits.detect_memory_gb` is the memory the process may use. It is read from,
first match winning:

1. `STREAM_LIMITS_MEMORY_GB`;
2. the variables named in `STREAM_LIMITS_MEMORY_VARS` (a size such as `16gb`);
3. the cgroup memory limit, the lowest one on the path to the root, so a scheduler's limit on the
   job is found even though the process sits in a child group;
4. PBS's own record of the job, `qstat -f $PBS_JOBID` (`Resource_List.mem`, else `vmem`), for
   sites that set no memory variable and enforce no cgroup;
5. `SLURM_MEM_PER_NODE`, or `SLURM_MEM_PER_CPU` times the CPUs;
6. the machine's physical memory.

Use it instead of asking the operating system how much memory there is: on a shared node that
number is the node's, and a guard of "a quarter of the memory" then allows many times what the
job has. [Schedulers and containers](schedulers.md) describes each source.

## Polars' memory budget

Polars 2.x has an out-of-core memory budget past which its streaming engine spills to disk
instead of growing. {py:func}`stream_limits.limit_memory` sets it from the memory limit.

A streaming sort of 1.9 GB of Parquet peaked at 6.2 GB with Polars' defaults, a multiple of the
data. With a 400 MB budget it peaked at 1.8 GB, in twice the time. The budget bounds what
Polars' operators hold, not the whole process: its Parquet readers hold about 1.5 GB more on
8 threads (less on fewer), and your own arrays are on top. The default budget is
`0.5 x (limit - 1.5 GB)`, at least 0.25 GB; change the share with `fraction=` or
`STREAM_LIMITS_POLARS_FRACTION`.

Some things to know:

- **It is read once, at import**, like the thread pool, so `limit_memory` must run before Polars
  is imported (a warning says so otherwise).
- **It is undocumented in Polars 2.x.** The names are those of Polars' binary
  (`POLARS_OOC_MEMORY_BUDGET_MB`, `POLARS_OOC_SPILL_DIR`, ...). So it is set only for Polars 2.0
  or later (read from the package metadata, without importing Polars), never over a budget you
  set yourself, and `STREAM_LIMITS_POLARS_SPILL=0` turns it off.
- **Where it spills** is Polars' temporary directory. On a cluster, make sure `$TMPDIR` is the
  job's scratch space, or give `spill_dir=` or `STREAM_LIMITS_SPILL_DIR`.
- **How much it helps depends on the code.** Code that sorts or joins a large Parquet scan
  through Polars' streaming engine gains the most. Code that already works in bounded pieces
  gains little: in hdfe-stream the peak of a 3-million-row fit did not change, and in dpd-stream
  on 4.8 million rows it fell from 2.0 to 1.6 GB, with identical results and no slowdown.

## Choosing between a fast path and a smaller one

{py:class}`stream_limits.MemoryBudget` is for code that has a fast path that needs a lot of
memory and a slower one that needs less, or an allocation that cannot shrink and should be
refused up front instead of at the operating system's kill.

> headroom = limit x (1 - reserve) - memory the process holds now

```python
from stream_limits import MemoryBudget

budget = MemoryBudget(limit_gb=None, reserve=None)   # defaults: the job's allocation, 0.15
need_gb = nnz * 12 / 1e9                              # the exact size of the allocation
if budget.fits(need_gb):
    build_matrix()
else:
    stream_instead()
```

- The **limit** defaults to {py:func}`stream_limits.detect_memory_gb`.
- The **reserve** is the share of the limit held back for what the code does not model:
  allocator slack, transient copies, pages a cgroup charges for files. The default is 0.15
  (`STREAM_LIMITS_MEMORY_RESERVE`). Lower it to run closer to the limit; raise it for a job that
  must not be killed.
- `headroom_gb(upcoming_gb)` subtracts allocations you know are coming later in the same phase,
  and `fits(need, upcoming, share)` asks about a share of the headroom (half, for per-thread
  copies that must leave room for others).
- It reads the process's resident memory (one file read on Linux; 0.5 GB is assumed where it
  cannot be read), so the answer reflects what the process holds *at that moment*.

Ask whether your **exact** allocation fits, not whether an estimate's upper bound does:
estimates are pessimistic by design, and refusing a job for an estimate's pessimism helps no one.

## Measuring what was used

{py:func}`stream_limits.peak_rss_gb` and {py:func}`stream_limits.reset_peak_rss` read the
kernel's own high-water mark of the process's resident memory. Nothing is polled, so it costs
nothing.

```python
stream_limits.reset_peak_rss()
fit_the_model()
print(stream_limits.peak_rss_gb())    # GB, the peak during the call
```

On Linux the counter is reset by writing to `/proc/self/clear_refs`, so a second measurement in
the same process reports its own peak; on macOS and Windows the system's counter cannot be
reset, so it is the process's lifetime peak (a second measurement is the largest of both). Child
processes are not included. {py:func}`stream_limits.current_rss_gb` is the memory held now.
