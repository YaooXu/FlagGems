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
from benchmark.reference import reference_report, validate_reference_options

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
    report = reference_report(config.reference_records)
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
    assert reference_report(config.reference_records)["status"] == "FAILED"
    assert not config.executed_case_ids


def test_unsupported_custom_runner_cannot_execute_candidate(runner):
    _, config, _ = runner
    class Custom(base.Benchmark):
        def run(self):
            forbidden()
    with pytest.raises(pytest.skip.Exception):
        Custom("custom", torch_op=forbidden)
    assert reference_report(config.reference_records, exitstatus=1)["status"] == "UNSUPPORTED"

@pytest.mark.parametrize("name", ["override", "override_config", "preflight_only", "profile_only", "list_cases", "query", "parallel", "numprocesses"])
def test_conflicting_modes_are_rejected(name):
    config = SimpleNamespace(option=SimpleNamespace(reference_only=True, **{name: True}))
    with pytest.raises(pytest.UsageError, match="reference-only"):
        validate_reference_options(config)

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

def test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness():
    report = reference_report([
        {"nodeid": "test", "case_id": "first", "status": "PASSED", "count": 1},
        {"nodeid": "test", "status": "SKIP", "pytest_phase": "call", "reason": "source condition"},
    ])
    assert report["status"] == "ALL_SKIP"
    assert report["records"][0]["count"] == 1


@pytest.mark.parametrize("reference_only", [False, True])
def test_reference_mode_cannot_run_correctness_pytest(runner, reference_only):
    from pathlib import Path
    _, config, _ = runner
    config.reference_only = reference_only
    item = SimpleNamespace(path=Path(conftest.__file__).parents[1] / "tests/test_negative.py")
    options = SimpleNamespace(getoption=lambda _: None)
    if reference_only:
        with pytest.raises(pytest.UsageError, match="benchmark only"):
            conftest.pytest_collection_modifyitems(None, options, [item])
    else:
        conftest.pytest_collection_modifyitems(None, options, [item])


@pytest.mark.parametrize("reference_only", [False, True])
@pytest.mark.parametrize("level", [None, "core", "comprehensive"])
def test_reference_only_defaults_to_core_without_changing_normal_defaults(monkeypatch, reference_only, level):
    from _pytest.config.argparsing import Parser
    parser = Parser()
    conftest.pytest_addoption(parser)
    argv = ["--reference-only"] if reference_only else []
    if level is not None:
        argv.extend(["--level", level])
    options = parser.parse(argv)
    config = SimpleNamespace(
        option=options, addinivalue_line=lambda *a: None, getini=lambda _: [],
        getoption=lambda key: getattr(options, key.lstrip("-").replace("-", "_")),
        hook=SimpleNamespace(pytest_flaggems_profile_scope=lambda **kw: None),
    )
    monkeypatch.setattr(conftest, "Config", conftest.Config)
    monkeypatch.setattr(conftest, "REPORT_FILE", conftest.REPORT_FILE)
    monkeypatch.setattr(conftest, "apply_overrides_from_args", lambda _: None)
    conftest.pytest_configure(config)
    assert conftest.Config.bench_level.value == (level or ("core" if reference_only else "comprehensive"))
