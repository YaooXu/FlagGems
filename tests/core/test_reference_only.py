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

"""Reference-only contract tests using CPU tensors and fake device operations."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmark import base, conftest
from benchmark.cases import BenchmarkCaseSpec
from flag_gems.testing.reference import (
    reference_call,
    reference_execution,
    reference_only,
    reference_report,
    validate_reference_options,
)

pytest_plugins = ["pytester"]


def forbidden(*args, **kwargs):
    raise AssertionError("candidate, timing or profiling must not run")


@pytest.fixture
def runner(monkeypatch):
    config = conftest.BenchConfig()
    config.reference_only = True
    config.current_nodeid = "benchmark/test_op.py::test_op"
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(conftest, "Config", config)
    events = []
    monkeypatch.setattr(base, "torch_device_fn", SimpleNamespace(synchronize=lambda: events.append("sync")))
    bench = base.Benchmark("op", torch_op=lambda value: events.append(value), gems_op=forbidden)
    monkeypatch.setattr(bench, "init_user_config", lambda: None)
    monkeypatch.setattr(bench, "supports_cases", lambda: True)
    cases = [BenchmarkCaseSpec(f"case-{i}", i, "float32", {}) for i in range(3)]
    monkeypatch.setattr(bench, "_collect_cases", lambda: cases)
    monkeypatch.setattr(bench, "build_inputs", lambda case: (case.ordinal,))
    monkeypatch.setattr(bench, "unpack_to_args_kwargs", lambda value: (value, {}))
    monkeypatch.setattr(bench, "_candidate_call", forbidden)
    monkeypatch.setattr(bench, "_time_callable", forbidden)
    return bench, config, events


@pytest.mark.parametrize("selected,expected", [(None, [0, 1, 2]), (["case-1"], [1])])
def test_benchmark_runs_original_baseline_once(runner, selected, expected):
    bench, config, events = runner
    assert bench.run(case_ids=selected) == [f"case-{i}" for i in expected]
    assert events == [x for i in expected for x in (i, "sync")]
    assert all(r["count"] == 1 and r["status"] == "PASSED" for r in config.reference_records)
    assert all("latency" not in r and "speedup" not in r for r in config.reference_records)


def test_backward_reference_uses_original_grad_semantics(runner, monkeypatch):
    bench, _, events = runner
    bench.is_backward = True
    bench.torch_op = lambda value: value.square()
    monkeypatch.setattr(bench, "build_inputs", lambda case: (torch.tensor([2.0], requires_grad=True),))
    original = torch.autograd.grad
    gradients = []
    def grad(*args, **kwargs):
        result = original(*args, **kwargs)
        gradients.extend(result)
        return result
    monkeypatch.setattr(torch.autograd, "grad", grad)
    bench.run(case_ids=["case-0"])
    assert len(gradients) == 1
    assert events == ["sync"]


def test_skip_native_is_not_a_pass_or_failure(runner):
    bench, config, events = runner
    config.skip_native = True
    config.native_baseline_skip_reason = "original vendor condition"
    assert bench.run() == []
    assert not events
    report = reference_report("timing", config.reference_records)
    assert report["status"] == "ALL_SKIP"
    assert {r["reason"] for r in report["records"]} == {"original vendor condition"}


@pytest.mark.parametrize("failure", ["input", "reference", "sync"])
def test_reference_failure_is_not_passed(runner, monkeypatch, failure):
    bench, config, _ = runner
    def broken(*a):
        raise RuntimeError("original failure")
    if failure == "input":
        monkeypatch.setattr(bench, "build_inputs", broken)
    elif failure == "reference":
        bench.torch_op = broken
    else:
        monkeypatch.setattr(base.torch_device_fn, "synchronize", broken)
    with pytest.raises(RuntimeError, match="original failure"):
        bench.run()
    assert reference_report("timing", config.reference_records)["status"] == "FAILED"
    assert not config.executed_case_ids


def test_unsupported_custom_runner_cannot_execute_candidate(runner):
    _, config, _ = runner
    class Custom(base.Benchmark):
        def run(self):
            forbidden()
    with pytest.raises(pytest.skip.Exception):
        Custom("custom", torch_op=forbidden)
    assert reference_report("timing", config.reference_records, exitstatus=1)["status"] == "UNSUPPORTED"


def test_reference_helper_preserves_normal_result_and_multiple_calls():
    assert not reference_only()
    assert reference_call(lambda value: value + 1, 2) == 3
    syncs = []
    with reference_execution(lambda: syncs.append(1)) as calls:
        assert reference_only()
        assert reference_call(lambda value: value + 1, 2) == 3
        assert reference_call(lambda: "second") == "second"
    assert not reference_only()
    assert syncs == [1, 1] and [r["status"] for r in calls] == ["PASSED", "PASSED"]


@pytest.fixture
def accuracy_pytest(pytester):
    pytester.makeini("[pytest]\n")
    pytester.makeconftest('''
from tests.conftest import *
from types import SimpleNamespace
import tests.conftest as source
source.torch_device_fn = SimpleNamespace(synchronize=lambda: None)
''')
    return pytester


@pytest.mark.parametrize("case,expected,exitcode", [
    ("multiple", "PASSED", 0), ("missing_call", "FAILED", 1),
    ("unsupported", "UNSUPPORTED", 1), ("skip", "ALL_SKIP", 0),
    ("failure", "FAILED", 1), ("teardown", "FAILED", 1),
])
def test_accuracy_pytest_reports_actual_reference_execution(accuracy_pytest, case, expected, exitcode):
    p = accuracy_pytest
    code = '''
import pytest
from flag_gems.testing.reference import reference_call, reference_only
@pytest.fixture
def teardown():
    yield
    raise RuntimeError("teardown failure")
'''
    marker = "" if case == "unsupported" else "@pytest.mark.reference_only\n"
    if case == "skip":
        marker += '@pytest.mark.skip(reason="source condition")\n'
    code += marker + 'def test_original(' + ('teardown' if case == 'teardown' else '') + '):\n'
    if case in {"skip", "unsupported"}:
        code += '    raise AssertionError("must not execute")\n'
    elif case == "missing_call":
        code += '    return\n'
    elif case == "failure":
        code += '    reference_call(lambda: 1 / 0)\n'
    else:
        code += '''    assert reference_only()
    reference_call(lambda: "first")
    if not reference_only():
        raise AssertionError("candidate executed")
    reference_call(lambda: "second")
'''
    p.makepyfile(test_original=code)
    result = p.runpytest_subprocess("-q", "--reference-only", "--output", "reference.json")
    assert result.ret == exitcode
    report = json.loads((p.path / "reference.json").read_text())
    assert report["schema_version"] == "flaggems.reference/v1"
    assert report["phase"] == "correctness" and report["status"] == expected
    if case == "multiple":
        assert len(report["records"][0]["reference_calls"]) == 2


@pytest.mark.parametrize("option", ["--override", "--override-config"])
def test_reference_only_rejects_candidate_injection_before_import(accuracy_pytest, option):
    p = accuracy_pytest
    p.makepyfile(test_original="def test_original(): pass")
    result = p.runpytest_subprocess("--reference-only", option, "must-not-be-loaded")
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize("name", ["preflight_only", "profile_only", "list_cases", "query", "parallel", "numprocesses"])
def test_conflicting_modes_are_rejected(name):
    config = SimpleNamespace(option=SimpleNamespace(reference_only=True, **{name: True}))
    with pytest.raises(pytest.UsageError, match="reference-only"):
        validate_reference_options(config)


def test_mixed_accuracy_and_timing_reports_are_rejected():
    config = SimpleNamespace(option=SimpleNamespace(reference_only=True))
    assert validate_reference_options(config, "correctness")
    with pytest.raises(pytest.UsageError, match="separate"):
        validate_reference_options(config, "timing")


@pytest.mark.parametrize("operator,expected", [("negative", 1), ("rsqrt", 1), ("rsqrt_", 1), ("addmm", 2)])
@pytest.mark.parametrize("cpu_reference,fp64", [(False, False), (False, True), (True, False)])
def test_adapted_pytests_preserve_references_without_candidate(monkeypatch, operator, expected, cpu_reference, fp64):
    import importlib
    import flag_gems
    from tests import accuracy_utils

    module = importlib.import_module("tests.test_" + ("rsqrt" if operator == "rsqrt_" else operator))
    monkeypatch.setattr(flag_gems, "device", "cpu")
    monkeypatch.setattr(flag_gems, "vendor_name", "nvidia")
    monkeypatch.setattr(flag_gems, "use_gems", forbidden)
    monkeypatch.setattr(flag_gems, "addmm", forbidden)
    monkeypatch.setattr(accuracy_utils, "TO_CPU", cpu_reference)
    monkeypatch.setattr(accuracy_utils, "fp64_is_supported", fp64)
    monkeypatch.setattr(accuracy_utils, "gems_assert_close", forbidden)
    monkeypatch.setattr(accuracy_utils, "gems_assert_equal", forbidden)
    seen = []
    def observe(function, *args, **kwargs):
        seen.append(args[0].dtype)
        return reference_call(function, *args, **kwargs)
    monkeypatch.setattr(module, "reference_call", observe)
    with reference_execution(lambda: None) as calls:
        if operator == "addmm":
            module.test_addmm(None, 2, 3, 4, 0.5, torch.float32, False)
        else:
            getattr(module, "test_" + operator)((2, 3), torch.float32)
    assert len(calls) == expected and all(r["status"] == "PASSED" for r in calls)
    dtype = torch.float64 if operator != "negative" and (fp64 or cpu_reference) else torch.float32
    assert seen == [dtype] * expected


def test_reference_report_replaces_stale_data_and_preserves_case_skips(runner, tmp_path, monkeypatch):
    bench, config, _ = runner
    output = tmp_path / "reference.json"
    output.write_text('{"stale":true}')
    monkeypatch.setattr(conftest, "REPORT_FILE", str(output))
    config.skip_native = True
    config.native_baseline_skip_reason = "unsupported by source"
    config.case_ids = ["case-1"]
    bench.run()
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK, config=SimpleNamespace(
        pluginmanager=SimpleNamespace(get_plugin=lambda _: None)))
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.OK
    conftest.pytest_terminal_summary(None, session.exitstatus, None)
    report = json.loads(output.read_text())
    assert report["status"] == "ALL_SKIP" and "stale" not in report
    assert report["records"][0]["case_id"] == "case-1"


def test_reference_unknown_case_selection_fails(runner):
    bench, config, events = runner
    config.case_ids = ["not-a-case"]
    bench.run()
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK, config=SimpleNamespace(
        pluginmanager=SimpleNamespace(get_plugin=lambda _: None)))
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert not events


def test_reference_call_preserves_source_skip():
    with reference_execution(forbidden) as calls:
        with pytest.raises(pytest.skip.Exception):
            reference_call(lambda: pytest.skip("original condition"))
    assert calls[0]["status"] == "SKIP"
    assert not reference_only()


def test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness():
    report = reference_report("timing", [
        {"nodeid": "test", "case_id": "first", "status": "PASSED", "count": 1},
        {"nodeid": "test", "status": "SKIP", "pytest_phase": "call", "reason": "source condition"},
    ])
    assert report["status"] == "ALL_SKIP"
    assert report["records"][0]["count"] == 1


def test_normal_addmm_still_compares_both_candidate_results(monkeypatch):
    import flag_gems
    from tests import test_addmm

    monkeypatch.setattr(flag_gems, "device", "cpu")
    monkeypatch.setattr(flag_gems, "vendor_name", "nvidia")
    calls = []
    def candidate(*args, **kwargs):
        calls.append(True)
        return torch.addmm(*args, **kwargs)
    monkeypatch.setattr(flag_gems, "addmm", candidate)
    assert not reference_only()
    test_addmm.test_addmm(None, 2, 3, 4, 0.5, torch.float32, False)
    assert len(calls) == 2
