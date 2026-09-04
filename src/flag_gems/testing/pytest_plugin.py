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

"""Built-in pytest options for running one external FlagGems candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("FlagGems candidate code")
    group.addoption(
        "--candidate-code-path",
        "--candidate_code_path",
        dest="candidate_code_path",
        default=None,
        help=(
            "Load run() from this Python file as the implementation under test. "
            "The selected pytest session must resolve exactly one public operator."
        ),
    )
    group.addoption(
        "--candidate-report-path",
        "--candidate_report_path",
        dest="candidate_report_path",
        default=None,
        help="Write candidate identity and per-test/per-case call coverage as JSON.",
    )
    group.addoption(
        "--candidate-call-count-path",
        "--candidate_call_count_path",
        dest="candidate_call_count_path",
        default=None,
        help="Write the total candidate invocation count as plain text.",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("candidate_code_path")
    report_path = config.getoption("candidate_report_path")
    call_count_path = config.getoption("candidate_call_count_path")
    if (report_path or call_count_path) and not path:
        raise pytest.UsageError(
            "candidate report options require --candidate-code-path."
        )
    if not path:
        return

    import flag_gems

    context = flag_gems.testing.candidate_code(path)
    try:
        context.__enter__()
    except Exception as exc:
        raise pytest.UsageError(
            f"failed to load --candidate-code-path {path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    config._flag_gems_candidate_context = context
    config._flag_gems_candidate_expected_nodeids = set()
    config._flag_gems_candidate_explicit_call_tracking = bool(
        report_path or call_count_path
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    if not hasattr(session.config, "_flag_gems_candidate_context"):
        return
    explicit = session.config._flag_gems_candidate_explicit_call_tracking
    rootpath = session.config.rootpath

    def is_benchmark_item(item: pytest.Item) -> bool:
        try:
            relative = item.path.relative_to(rootpath)
        except ValueError:
            return False
        return relative.parts[:1] == ("benchmark",)

    has_correctness = any(not is_benchmark_item(item) for item in session.items)
    track_calls = explicit or has_correctness

    import flag_gems

    flag_gems.testing.set_candidate_call_tracking(track_calls)
    session.config._flag_gems_candidate_track_calls = track_calls


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
):
    outcome = yield
    report = outcome.get_result()
    expected = getattr(
        item.config, "_flag_gems_candidate_expected_nodeids", None
    )
    if expected is not None and report.when == "call" and not report.skipped:
        expected.add(report.nodeid)


def _candidate_report(config: pytest.Config) -> dict[str, Any] | None:
    if not hasattr(config, "_flag_gems_candidate_context"):
        return None
    import flag_gems

    expected = tuple(
        sorted(config._flag_gems_candidate_expected_nodeids)
    )
    return flag_gems.testing.candidate_report(expected)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    report = _candidate_report(session.config)
    if report is None:
        return

    errors = []
    if report["operator"] is None:
        errors.append("the selected tests never resolved a public gems_op")
    if report["call_tracking"]:
        if report["total_calls"] == 0:
            errors.append("the candidate run() function was never called")
        if report["missing_nodeids"]:
            preview = ", ".join(report["missing_nodeids"][:20])
            errors.append("selected tests bypassed the candidate: " + preview)
    session.config._flag_gems_candidate_errors = errors

    output = session.config.getoption("candidate_report_path")
    if output:
        import flag_gems

        flag_gems.testing.write_candidate_report(
            output,
            tuple(sorted(session.config._flag_gems_candidate_expected_nodeids)),
        )
    call_count_output = session.config.getoption("candidate_call_count_path")
    if call_count_output:
        path = Path(call_count_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(report["total_calls"]), encoding="utf-8")

    if errors and session.exitstatus in {
        pytest.ExitCode.OK,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    }:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    errors = getattr(
        terminalreporter.config, "_flag_gems_candidate_errors", None
    )
    if not errors:
        return
    terminalreporter.section("FlagGems candidate code errors", red=True)
    for error in errors:
        terminalreporter.write_line(f"- {error}", red=True)


def pytest_unconfigure(config: pytest.Config) -> None:
    context = getattr(config, "_flag_gems_candidate_context", None)
    if context is not None:
        context.__exit__(None, None, None)


__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_collection_finish",
    "pytest_runtest_makereport",
    "pytest_sessionfinish",
    "pytest_terminal_summary",
    "pytest_unconfigure",
]
