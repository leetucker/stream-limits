"""Internal pre-exec attachment helper, deliberately standard-library-only."""

import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("internal usage: PARENT -- COMMAND ...")
    path = Path(sys.argv[1])
    (path / "cgroup.procs").write_text(str(os.getpid()))
    command = sys.argv[3:]
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
