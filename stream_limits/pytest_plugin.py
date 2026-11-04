"""Explicit pytest plugin: pytest_plugins = ['stream_limits.pytest_plugin'].

Not auto-loaded, and never imported by normal stream_limits users.
"""

import os
from pathlib import Path

import pytest

from .launcher import run_limited
from .testing import verify_memory_limit


@pytest.fixture
def memory_runner():
    """The common run_limited callable; each test supplies its own command/mode."""
    return run_limited


@pytest.fixture(scope="session")
def _configured_memory_parent():
    parent = os.environ.get("STREAM_LIMITS_TEST_CGROUP_PARENT")
    if parent:
        verify_memory_limit(parent)
        return Path(parent)
    return None


@pytest.fixture(scope="session")
def verified_memory_parent(_configured_memory_parent):
    """Skip kernel-only checks unless a delegate was explicitly configured."""
    if _configured_memory_parent is None:
        pytest.skip(
            "set STREAM_LIMITS_TEST_CGROUP_PARENT to a delegated Linux memory cgroup"
        )
    return _configured_memory_parent


@pytest.fixture(scope="session")
def resource_enforcement(_configured_memory_parent):
    """Portable resource-test policy, explicitly reported to each test.

    Uses a sampled watchdog on all supported platforms. A configured delegated
    Linux cgroup is verified and used instead; an unusable explicit setting fails.
    """
    if _configured_memory_parent is not None:
        return {"mode": "hard", "cgroup_parent": _configured_memory_parent}
    return {"mode": "watchdog"}


@pytest.fixture
def resource_runner(resource_enforcement):
    """Constrain resource tests on every supported OS without skipping them.

    Takes run_limited's arguments except mode/cgroup_parent (e.g. memory_gb,
    timeout, env, threads)."""

    def run(command, **kwargs):
        if "mode" in kwargs or "cgroup_parent" in kwargs:
            raise ValueError(
                "resource_runner selects its mode explicitly via resource_enforcement"
            )
        return run_limited(command, **resource_enforcement, **kwargs)

    return run


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "resource: explicit constrained-resource subprocess test"
    )
    config.addinivalue_line(
        "markers",
        "kernel_memory: requires an explicitly delegated Linux memory controller",
    )
