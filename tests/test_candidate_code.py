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

import hashlib
from types import SimpleNamespace

import pytest

import flag_gems
from flag_gems.testing import pytest_plugin as candidate_pytest_plugin


def test_candidate_code_loads_run_binds_once_and_restores(tmp_path):
    source = b"def run(value):\n    return ('candidate', value)\n"
    candidate = tmp_path / "main.py"
    candidate.write_bytes(source)
    default = lambda value: ("default", value)

    assert flag_gems.testing.resolve_gems_op("unit_candidate", default) is default
    with flag_gems.testing.candidate_code(str(candidate)):
        resolved = flag_gems.testing.resolve_gems_op(
            "unit_candidate", default
        )
        assert resolved("value") == ("candidate", "value")
        assert (
            flag_gems.testing.gems_op_source("unit_candidate", resolved)
            == "override"
        )
        report = flag_gems.testing.candidate_report()
        assert report["operator"] == "unit_candidate"
        assert report["source_sha256"] == hashlib.sha256(source).hexdigest()
        assert report["total_calls"] == 1

    assert flag_gems.testing.resolve_gems_op("unit_candidate", default) is default


def test_candidate_report_tracks_pytest_node_and_benchmark_case(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value\n", encoding="utf-8")
    nodeid = "benchmark/test_example.py::test_example[param]"
    case_id = nodeid + "::core::float16::0"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", nodeid + " (call)")

    with flag_gems.testing.candidate_code(str(candidate)):
        resolved = flag_gems.testing.resolve_gems_op(
            "tracked_operator", lambda value: value
        )
        with flag_gems.testing.gems_op_case("tracked_operator", case_id):
            assert resolved("value") == "value"
        report = flag_gems.testing.candidate_report((nodeid, "missing-node"))

    assert report["missing_nodeids"] == ["missing-node"]
    assert report["records"] == [
        {"nodeid": nodeid, "case_id": case_id, "count": 1}
    ]


def test_candidate_code_keeps_sibling_imports_available(tmp_path):
    (tmp_path / "_flag_gems_candidate_test_helper.py").write_text(
        "def apply(value):\n    return value + 1\n", encoding="utf-8"
    )
    candidate = tmp_path / "main.py"
    candidate.write_text(
        "def run(value):\n"
        "    from _flag_gems_candidate_test_helper import apply\n"
        "    return apply(value)\n",
        encoding="utf-8",
    )

    with flag_gems.testing.candidate_code(str(candidate)):
        resolved = flag_gems.testing.resolve_gems_op("unit_sibling", lambda x: x)
        assert resolved(1) == 2


def test_candidate_code_can_disable_call_tracking_for_timing(tmp_path):
    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value\n", encoding="utf-8")

    with flag_gems.testing.candidate_code(
        str(candidate), track_calls=False
    ):
        resolved = flag_gems.testing.resolve_gems_op(
            "timed_operator", lambda value: value
        )
        assert resolved("value") == "value"
        report = flag_gems.testing.candidate_report()

    assert resolved.__name__ == "run"
    assert report["call_tracking"] is False
    assert report["total_calls"] == 0


def test_benchmark_resolves_candidate_without_dispatcher_mutation(tmp_path):
    from benchmark import base

    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value + 1\n", encoding="utf-8")
    default = lambda value: value
    benchmark = base.Benchmark(
        op_name="benchmark_candidate",
        torch_op=default,
        gems_op=default,
    )

    with flag_gems.testing.candidate_code(str(candidate), track_calls=False):
        resolved, source = benchmark._resolve_direct_gems_op()
        assert resolved(1) == 2
        assert source == "override"

    resolved, source = benchmark._resolve_direct_gems_op()
    assert resolved is default
    assert source == "default"


def test_pytest_plugin_disables_tracking_for_benchmark_only_session(tmp_path):
    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value\n", encoding="utf-8")
    root = tmp_path / "repo"
    benchmark_path = root / "benchmark/test_example.py"
    session = SimpleNamespace(
        config=SimpleNamespace(
            rootpath=root,
            _flag_gems_candidate_context=object(),
            _flag_gems_candidate_explicit_call_tracking=False,
        ),
        items=[SimpleNamespace(path=benchmark_path)],
    )

    with flag_gems.testing.candidate_code(str(candidate)):
        candidate_pytest_plugin.pytest_collection_finish(session)
        resolved = flag_gems.testing.resolve_gems_op(
            "timed_operator", lambda value: value
        )
        assert resolved("value") == "value"
        report = flag_gems.testing.candidate_report()

    assert report["call_tracking"] is False
    assert report["total_calls"] == 0


def test_candidate_code_rejects_a_second_operator(tmp_path):
    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value\n", encoding="utf-8")

    with flag_gems.testing.candidate_code(str(candidate)):
        flag_gems.testing.resolve_gems_op("first_operator", lambda x: x)
        with pytest.raises(RuntimeError, match="only one public operator"):
            flag_gems.testing.resolve_gems_op("second_operator", lambda x: x)


def test_candidate_code_requires_callable_run(tmp_path):
    candidate = tmp_path / "main.py"
    candidate.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="callable run"):
        with flag_gems.testing.candidate_code(str(candidate)):
            pass


def test_candidate_code_rejects_explicit_override_conflict(tmp_path):
    candidate = tmp_path / "main.py"
    candidate.write_text("def run(value):\n    return value\n", encoding="utf-8")

    with flag_gems.testing.candidate_code(str(candidate)):
        with flag_gems.testing.override_gems_op("conflict", lambda x: x):
            with pytest.raises(RuntimeError, match="cannot be combined"):
                flag_gems.testing.resolve_gems_op("conflict", lambda x: x)
