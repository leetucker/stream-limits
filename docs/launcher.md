# Running a program under a memory limit

The launcher runs a program (a Python or R script, a test suite, any executable)
in a child process, watches its memory, and stops it if it goes over a limit you
set. When the program ends it tells you why it ended and how much memory it used.

It is an optional extra, which adds psutil and nothing else:

```bash
pip install 'stream-limits[launcher]'
```

Nothing else in stream-limits changes: `import stream_limits` still needs only
the standard library, and the launcher imports no numerical library itself.

## A first example

```bash
python -m stream_limits.launcher --mode watchdog --memory-gb 4 --threads 2 \
    --timeout 600 -- python analysis.py
```

This runs `python analysis.py` with 2 threads for every numerical library, stops
it if it and the processes it starts use more than 4 GB together or run longer
than 10 minutes, and prints a JSON summary: the status, the exit code, the time,
the peak memory, and the end of the program's output. The exit code is the
program's own (124 on a timeout).

The same from Python:

```python
import sys
from stream_limits.launcher import run_limited

result = run_limited([sys.executable, "analysis.py"],
                     mode="watchdog", memory_gb=4, threads=2, timeout=600)
print(result.status, result.os_peak_rss_bytes / 1e9, "GB")
```

The command is a list of arguments, run without a shell. GB are decimal
(1 GB = 10^9 bytes).

## Three ways to limit

| Mode | Platforms | What it does |
|---|---|---|
| `watchdog` | Linux, macOS, Windows | Checks memory about every 10 ms and stops the program when it passes the limit |
| `hard` | Linux, with a cgroup set up for it | The kernel enforces the limit |
| `monitor` | Linux, macOS, Windows | No limit: measures, with an optional timeout |

**The watchdog** adds up the memory of the program and every process it has
started, and stops them all when the total passes the limit. It counts
*private* memory: memory that is not the pages of a file. A program that
memory-maps a large file (a scratch array on disk, a mapped Parquet file) has
the file's pages in its resident memory (RSS), but the kernel can drop them and
read them back when memory runs short, so they are not counted against the
limit. hdfe-stream is such a program: on a 20-million-row panel its RSS reached
2.3 GB while its private memory stayed at 1.2 GB, the same as on 2.6 million
rows. On Linux private memory is anonymous plus shared memory; on Windows it is
private bytes (committed memory, which can be more than RSS). macOS has no cheap
equivalent, so there the watchdog counts RSS. `--memory-measure rss`
(`memory_measure="rss"`) counts every resident page on every platform, and the
result's `memory_measure` says which was used. Because it samples,
it can overshoot a little before it acts, and a peak shorter than the interval
can slip between samples. In practice it is tight: on Linux, with a 1.5 GB limit,
a fast 3 GB NumPy allocation, a 2.5 GB spike lasting 20 ms and a grandchild
process allocating 3 GB were each stopped at 1.51-1.52 GB. Memory shared
between processes can be counted more than once. If the operating system denies
it access to a process's memory, the watchdog stops the program and reports
`monitoring_failed` rather than claim it enforced the limit.

**Hard limits** leave nothing to sampling: the kernel will not let the program
use more, and kills it if it tries. They need a Linux cgroup that you are
allowed to manage (below). `hard` is the default mode, so that asking for a hard
limit where none is possible fails before the program starts, instead of quietly
running under the watchdog. macOS and Windows have no hard mode.

## Results

`run_limited` returns a `RunResult` (`as_dict()` turns it into JSON). Its
`status` is one of:

| Status | Meaning |
|---|---|
| `ok` | The program finished with exit code 0 |
| `command_failed` | It finished with another exit code |
| `timeout` | It ran past `timeout` and was stopped |
| `rss_limit_exceeded` | The watchdog stopped it (by the measure in `memory_measure`, despite the name) |
| `monitoring_failed` | The watchdog could not read its memory and stopped it |
| `kernel_oom` | The kernel killed a process for going over a hard limit |

`kernel_oom` comes from the kernel's own counter, not from guessing at a killed
process. It is reported even when the process killed was a child the program
started and the program itself carried on and exited with 0, so check `status`
and not only the exit code.

Three peaks are reported:

- `peak_private_bytes`: the largest sampled total of private memory over all the
  program's processes (`None` on macOS). This is the one to use for a program
  that memory-maps files, and the one the watchdog limits.
- `os_peak_rss_bytes`: the largest RSS of any single process, as the operating
  system records it. On Linux and macOS it is exact, and covers the program and
  every process it started and waited for (not one it left running in the
  background). On Windows it is read while sampling. It counts mapped file
  pages. Use it when a test needs a reliable peak of a program that maps no
  large files, or on macOS.
- `peak_rss_bytes`: the largest sampled total RSS over all the program's
  processes at once. A program that ends before the first sample reports 0.

With a hard limit, `result.kernel` adds the kernel's own figures: the limit, any
smaller limit set higher up, its peak (which counts file cache and kernel
memory, so it differs from the two above), and its event counters.

## Output, threads and the environment

By default the program's output is saved to temporary files and the last
megabyte of each stream is returned in the result (`max_output_bytes=`). To see
it as it runs, as when wrapping a whole test run, pass `--no-capture`
(`capture_output=False`). The program then writes to the terminal and the JSON
summary goes to stderr; `--report FILE` also saves the summary to a file:

```bash
python -m stream_limits.launcher --mode watchdog --memory-gb 4 --threads 2 \
    --no-capture --report memory.json -- python -m pytest -q
```

`--threads N` (`threads=N`) sets `STREAM_LIMITS_MAX_THREADS` and the thread
variables of Polars, numba, OpenMP, the BLAS libraries, numexpr, rayon and Julia
in the program's environment. `stream_limits.launcher.thread_env(N, env)`
returns that environment. Without `threads`, no variable is added: `env`
replaces the environment as in `subprocess`, and the launcher's own environment
is never changed.

On a timeout or when a limit is hit, the program and the processes it started
are stopped. On Linux and macOS the launcher also stops anything left in the
program's process group, and with a hard limit anything left in its cgroup. On
Windows, a process that escapes before it is first sampled can survive.

## Setting up hard limits

Hard limits use a Linux cgroup with the memory controller, which an
administrator (or you, with root) sets up once and lets you manage. The launcher
never uses sudo and never changes a cgroup it did not create: for each run it
creates a fresh group inside the one you give it, sets the limit there, starts
the program inside it (the launcher itself stays outside the limit), and removes
the group afterwards. It is a resource limit, not a security sandbox.

`python -m stream_limits.launcher --capabilities` and `cat /proc/self/cgroup`
show what the machine has.

**cgroup v1** (WSL2, older distributions), once, as root:

```bash
sudo mkdir /sys/fs/cgroup/memory/stream-limits-test
sudo chown -R "$USER" /sys/fs/cgroup/memory/stream-limits-test
export STREAM_LIMITS_TEST_CGROUP_PARENT=/sys/fs/cgroup/memory/stream-limits-test
```

The group is gone after a reboot.

**cgroup v2 with systemd**: start a shell in a user scope that you may manage:

```bash
systemd-run --user --scope -p Delegate=yes -p MemoryMax=8G bash
export STREAM_LIMITS_TEST_CGROUP_PARENT=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
```

On v2 the group must have `memory` in its `cgroup.subtree_control` and no
processes of its own, so the shell may first need to move into a child group.

These recipes are standard, but stream-limits' tests do not run them (they need
root or a systemd scope). Check a setup before relying on it:

```bash
python -c "from stream_limits.testing import verify_memory_limit as v; v('$STREAM_LIMITS_TEST_CGROUP_PARENT')"
```

This runs a small allocation that must succeed, checks that a program inside the
group sees the limit, and runs a 256 MiB allocation in a 64 MiB group that the
kernel must kill. It raises unless all three happen. (The group needs room for
64 MiB.)

Then:

```python
result = run_limited([sys.executable, "analysis.py"], mode="hard", memory_gb=4,
                     cgroup_parent="/sys/fs/cgroup/memory/stream-limits-test")
```

Before running anything, the launcher checks that the group really is a kernel
cgroup (not an ordinary directory with the same file names) and reads the limit
back. It turns swap off for the group (v2: `memory.swap.max=0`; v1: a
memory-plus-swap limit equal to the memory limit and `swappiness=0`), so the
limit cannot be dodged by swapping. On v2 the kernel stops the whole group
together when one process goes over. Older v2 kernels have no peak counter;
their kernel peak is then `None`.

The launcher's tests have exercised hard limits on a WSL2 host with cgroup v1.
cgroup v2, macOS and Windows rely on CI and have not been checked by hand.

## Memory tests in pytest

The plugin gives other packages' test suites a common way to run a test's work in
a child process under a memory limit. Install `stream-limits[testing]` and
register it in `conftest.py` (it is not loaded automatically):

```python
pytest_plugins = ["stream_limits.pytest_plugin"]
```

A memory test then looks like this:

```python
import sys
import pytest

@pytest.mark.resource
def test_fit_stays_within_memory(resource_runner, tmp_path):
    result = resource_runner([sys.executable, "fit_large_case.py", str(tmp_path)],
                             memory_gb=1.0, threads=2, timeout=120)
    assert result.status == "ok", result.as_dict()
```

`resource_runner` takes `run_limited`'s arguments except the mode. It uses the
watchdog on every platform, so these tests run on macOS and Windows too. When
`STREAM_LIMITS_TEST_CGROUP_PARENT` names a cgroup set up as above, it checks it
once per session with `verify_memory_limit` and uses a hard limit instead; a
setting that does not work fails the session rather than falling back.

```bash
pytest -m resource                                                    # watchdog
STREAM_LIMITS_TEST_CGROUP_PARENT=/sys/fs/cgroup/memory/stream-limits-test pytest -m resource
```

The other fixtures: `resource_enforcement` says which mode was chosen
(`{"mode": "watchdog"}` or `{"mode": "hard", "cgroup_parent": ...}`);
`memory_runner` is `run_limited` itself, for a test that picks its own mode; and
`verified_memory_parent` is the checked cgroup, for tests that need a hard limit
and skip without one. The plugin registers the markers `resource` and
`kernel_memory`.

**A memory test is only as good as its input.** A child process that imports
Polars, NumPy and numba already holds 0.2-0.4 GB, so a small input under a 1.5 GB
limit passes whatever the code does with it. To catch code that holds the data in
memory, give the test an input larger in memory than the limit less that
baseline, or compare peaks at N rows and at several times N:
`peak_private_bytes` where it is available, since a program that memory-maps
its scratch files has an RSS that grows with the data even when its private
memory does not.

Use the launcher only for tests about memory. Ordinary tests are better run in
the test process, as before.
