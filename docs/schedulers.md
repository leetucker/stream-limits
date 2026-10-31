# Schedulers and containers

`stream-limits` reads the job's allocation from the places schedulers and containers put it.
**`python -m stream_limits` prints what it found and where each number came from**; run it inside
a job (a batch job, or an interactive one) before relying on anything else.

```
$ python -m stream_limits
{
  "cpus": 8,
  "cpus_source": "NCPUS",
  "memory_gb": 17.18,
  "memory_source": "qstat Resource_List.mem",
  "env": {"NCPUS": "8", "PBS_JOBID": "123456.pbsserver"}
}
```

If a number is wrong or comes from `machine (no limit found)` / `physical memory`, see
[A site that does it differently](#a-site-that-does-it-differently).

## Where each limit comes from

First match wins, for each of the two limits.

### CPUs (threads)

| Order | Source | Notes |
|---|---|---|
| 1 | `STREAM_LIMITS_MAX_THREADS` | your own setting; never capped |
| 2 | variables named in `STREAM_LIMITS_CPU_VARS` | for sites with their own variable |
| 3 | `NCPUS` | PBS Pro exports the job's CPU count here |
| 4 | `PBS_NCPUS` | set by some PBS builds |
| 5 | `PBS_NUM_PPN`, `PBS_NP` | Torque: processors per node, total processors |
| 6 | `SLURM_CPUS_PER_TASK`, `SLURM_CPUS_ON_NODE` | Slurm |
| 7 | the cgroup CPU quota | `cpu.max` (v2) or `cpu.cfs_quota_us` (v1), the lowest on the path to the root; a quota of 2.5 CPUs gives 3 |
| 8 | `qstat -f $PBS_JOBID` | only if `PBS_JOBID` is set; see below |
| 9 | the process's CPU affinity | only if it is restricted below the machine's CPU count |
| — | `os.cpu_count()` | no limit found |

### Memory

| Order | Source | Notes |
|---|---|---|
| 1 | `STREAM_LIMITS_MEMORY_GB` | your own setting |
| 2 | variables named in `STREAM_LIMITS_MEMORY_VARS` | each holds a size such as `16gb`, `512mb`; a bare number is bytes |
| 3 | the cgroup memory limit | `memory.max` (v2) or `memory.limit_in_bytes` (v1), the lowest on the path to the root |
| 4 | `qstat -f $PBS_JOBID` | only if `PBS_JOBID` is set; see below |
| 5 | `SLURM_MEM_PER_NODE`, or `SLURM_MEM_PER_CPU` times the CPUs | megabytes |
| — | the machine's physical memory | no limit found |

## PBS

Environment variables differ between PBS versions and between sites. What is known:

- **PBS Pro** exports `NCPUS` to the job, and **no memory variable**. This was checked on one PBS
  Pro system, where `NCPUS` was set and nothing carried the memory request.
- **Torque** exports `PBS_NUM_PPN` and `PBS_NP`. These are taken from Torque's documentation and
  have not been checked on a Torque system here.
- `PBS_NODEFILE` is **not** used. It lists hosts, not CPUs: one line per CPU in Torque, but one
  line per chunk (or per MPI process) in PBS Pro, so counting its lines would silently cap a
  PBS Pro job at one thread.

### `qstat` as a fallback

When the variables above do not settle a limit, and `PBS_JOBID` is set, `stream-limits` asks PBS
what it recorded for the job:

```
qstat -f $PBS_JOBID
```

and reads the `Resource_List` attributes:

- **memory**: `Resource_List.mem`, else `Resource_List.vmem` (virtual memory, a looser limit);
- **CPUs**: `Resource_List.ncpus`, else the first chunk of `select` (`1:ncpus=8:mem=16gb`) or
  Torque's `nodes` (`2:ppn=4`), else `procs`.

Sizes carry PBS's suffixes (`b`, `kb`, `mb`, `gb`, `tb`, `pb`, and `w` words of 8 bytes, all
powers of 1024); a bare number is bytes.

How it is run: with a fixed argument list and no shell, after checking that the job id is made of
the characters a job id has, with a 10 second timeout, **once per job** (the answer serves both
limits and is cached), and never when a variable or a cgroup already settled the limit. If
`qstat` is missing, fails or reports nothing, it is skipped without an error.
`STREAM_LIMITS_QSTAT=0` turns it off.

Two things to know:

- **Several chunks.** For a job with more than one chunk, PBS reports the *total* over the
  chunks, but the Python process runs on one host. The CPU count from `qstat` is capped at the
  CPUs of the machine the process is on, and for memory you can set `STREAM_LIMITS_MEMORY_GB`
  to the chunk's share.
- **Job arrays and sub-jobs** need the id your site's `qstat -f` accepts; if the lookup returns
  nothing, set `STREAM_LIMITS_MEMORY_GB` and `STREAM_LIMITS_MAX_THREADS` in the job script.

## Slurm

`SLURM_CPUS_PER_TASK` and `SLURM_CPUS_ON_NODE` give the CPUs; `SLURM_MEM_PER_NODE` the memory
(in megabytes), or `SLURM_MEM_PER_CPU` times the CPUs when the job asked per CPU. Slurm sites
that enforce cgroups are also found through the cgroup limits.

## cgroups and containers

Containers and many schedulers limit a job with cgroups. The limit is usually on a *parent* group
(the job), not on the group the process sits in, so `stream-limits` reads the process's cgroup
from `/proc/self/cgroup` and takes the lowest limit on the path up to the root. Both cgroup v1
and v2 are supported. A cgroup that sets no limit (`max`) is skipped.

## A site that does it differently

Name the variable. Each of these settings is a comma-separated list of variable names, checked
before the built-in ones:

```bash
export STREAM_LIMITS_CPU_VARS=SITE_NCPUS        # holds an integer
export STREAM_LIMITS_MEMORY_VARS=SITE_MEMORY    # holds a size: 16gb, 512mb
```

or set the limit directly in the job script, which beats everything:

```bash
export STREAM_LIMITS_MAX_THREADS=$NCPUS
export STREAM_LIMITS_MEMORY_GB=16
```

If you find a variable that a scheduler sets and `stream-limits` does not read, please open an
issue with the output of `env | grep -i -E 'pbs|ncpus|mem|slurm'` from inside a job.

## A job script

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=04:00:00

cd $PBS_O_WORKDIR
python -m stream_limits      # shows what the job will be capped at
python my_analysis.py        # which starts with: import stream_limits; stream_limits.limit_threads()
```

With a package built on `stream-limits` (hdfe-stream, logit-stream, dpd-stream) the script needs no
`stream_limits` lines: the package applies the limits when it is imported.
