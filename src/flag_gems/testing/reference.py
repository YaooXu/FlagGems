# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit reference-only execution for pytest; never substitute a candidate."""

from contextlib import contextmanager
from contextvars import ContextVar


_active = ContextVar("flaggems_reference_execution", default=None)


def reference_only():
    """True only while pytest is executing an opted-in reference-only test."""
    return _active.get() is not None


@contextmanager
def reference_execution(synchronize):
    calls = []
    token = _active.set((calls, synchronize))
    try:
        yield calls
    finally:
        _active.reset(token)


def reference_call(function, *args, **kwargs):
    """Call the original reference, preserving its arguments, results and errors.

    Outside reference-only execution this is an ordinary function call. Inside,
    record completion only after device synchronization has succeeded.
    """
    active = _active.get()
    if active is None:
        return function(*args, **kwargs)
    calls, synchronize = active
    record = {"status": "FAILED"}
    calls.append(record)
    try:
        result = function(*args, **kwargs)
        synchronize()
    except BaseException as error:
        import pytest

        if isinstance(error, pytest.skip.Exception):
            record["status"] = "SKIP"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    record["status"] = "PASSED"
    return result


def add_reference_option(parser):
    # Both conftests may be loaded for a mixed correctness/benchmark collection.
    try:
        parser.addoption("--reference-only", action="store_true", default=False,
                         help="Run original references only; no candidate comparison or timing.")
    except ValueError:
        pass


def validate_reference_options(config, phase=None):
    if not getattr(config.option, "reference_only", False):
        return False
    import pytest

    previous = getattr(config, "_flaggems_reference_phase", None)
    if previous is not None and phase is not None and previous != phase:
        raise pytest.UsageError("--reference-only requires separate correctness and benchmark invocations")
    if phase is not None:
        config._flaggems_reference_phase = phase
    conflicts = [name for name in ("override", "override_config", "profile_only",
                 "preflight_only", "list_cases", "query", "parallel", "numprocesses")
                 if getattr(config.option, name, None)]
    if conflicts:
        raise pytest.UsageError("--reference-only cannot be combined with " +
                                ", ".join("--" + name.replace("_", "-") for name in conflicts))
    return True


def reference_report(phase, records, *, exitstatus=0):
    """Separate source skips, unsupported tests and real execution failures."""
    # pytest.skip aborts the entire original benchmark node, including any
    # remaining cases. Earlier calls remain evidence, not a completed node.
    skipped_nodes = {r.get("nodeid") for r in records
                     if r.get("pytest_phase") and r["status"] == "SKIP"}
    statuses = ["SKIP" if r["status"] == "PASSED" and r.get("nodeid") in skipped_nodes
                else r["status"] for r in records]
    if exitstatus not in (0, 1, 5) or "FAILED" in statuses:
        status = "FAILED"
    elif "UNSUPPORTED" in statuses:
        status = "UNSUPPORTED"
    elif exitstatus == 1:
        status = "FAILED"
    elif "PASSED" in statuses:
        status = "PASSED"
    elif statuses:
        status = "ALL_SKIP"
    else:
        status = "NO_CASES"
    return {"schema_version": "flaggems.reference/v1", "phase": phase,
            "status": status, "records": records}
