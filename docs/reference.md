# Reference

## Limiting

```{eval-rst}
.. module:: stream_limits

.. autofunction:: set_limits

.. autofunction:: limit_threads

.. autofunction:: limit_memory

.. autofunction:: apply_scheduler_defaults

.. autofunction:: thread_scope

.. autofunction:: set_openmp_wait_policy

.. autofunction:: polars_spills
```

## Finding the allocation

```{eval-rst}
.. autofunction:: detect_cpus

.. autofunction:: detect_memory_gb

.. autofunction:: describe
```

## A memory budget

```{eval-rst}
.. autoclass:: MemoryBudget
   :members: usable_gb, held_gb, headroom_gb, fits
```

## Measuring memory

```{eval-rst}
.. autofunction:: peak_rss_gb

.. autofunction:: reset_peak_rss

.. autofunction:: current_rss_gb
```

## Environment variables

| Variable | Meaning |
|---|---|
| `STREAM_LIMITS_MAX_THREADS` | the thread cap |
| `STREAM_LIMITS_MEMORY_GB` | the memory limit |
| `STREAM_LIMITS_CPU_VARS` | comma-separated names of variables holding a CPU count, checked before the built-in ones |
| `STREAM_LIMITS_MEMORY_VARS` | the same for memory; each holds a size (`16gb`, `512mb`; a bare number is bytes) |
| `STREAM_LIMITS_MEMORY_RESERVE` | `MemoryBudget`'s reserve (default 0.15) |
| `STREAM_LIMITS_POLARS_SPILL` | `0`: do not set Polars' out-of-core budget |
| `STREAM_LIMITS_POLARS_FRACTION` | the budget's share of the limit above the readers' floor (default 0.5) |
| `STREAM_LIMITS_SPILL_DIR` | where Polars spills (default its temp dir: on a cluster, check `$TMPDIR`) |
| `STREAM_LIMITS_QSTAT` | `0`: do not run `qstat` to find a PBS job's CPUs and memory |

`limit_threads` also sets these, unless you already have (`override=True` replaces them):
`POLARS_MAX_THREADS`, `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `NUMEXPR_MAX_THREADS`, `RAYON_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`,
`TF_NUM_INTEROP_THREADS`, `LOKY_MAX_CPU_COUNT`, `JULIA_NUM_THREADS`. `limit_memory` sets
`POLARS_OOC_MEMORY_BUDGET_MB` and, if asked, `POLARS_OOC_SPILL_DIR`.

## Optional launcher

```{eval-rst}
.. automodule:: stream_limits.launcher
   :members: RunResult, run_limited, capabilities, thread_env

.. automodule:: stream_limits.testing
   :members: verify_memory_limit, kernel_oom_proved
```
