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

import csv

import pytest

from . import ascend_timing, base, consts


def _write_op_statistics(path, rows):
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["OP Type", "Count", "Total Time(us)"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_collect_latency_sums_all_repeated_candidate_kernels(tmp_path):
    _write_op_statistics(
        tmp_path / "nested" / "op_statistic.csv",
        [
            {
                "OP Type": ascend_timing.L2_CACHE_CLEAR_KERNEL_NAME,
                "Count": "4",
                "Total Time(us)": "400",
            },
            {"OP Type": "candidate_a", "Count": "4", "Total Time(us)": "40"},
            {"OP Type": "candidate_b", "Count": "8", "Total Time(us)": "80"},
            {"OP Type": "unrelated", "Count": "3", "Total Time(us)": "999"},
        ],
    )

    assert ascend_timing._collect_latency_us(str(tmp_path), active_count=4) == 30.0


def test_collect_latency_rejects_missing_profiler_data(tmp_path):
    with pytest.raises(RuntimeError, match="op_statistic.csv"):
        ascend_timing._collect_latency_us(str(tmp_path), active_count=4)


def test_measure_latency_converts_microseconds_and_forwards_budgets(monkeypatch):
    events = []

    class FakeNpu:
        @staticmethod
        def synchronize():
            events.append("synchronize")

    class FakeTorch:
        npu = FakeNpu()

    monkeypatch.setattr(ascend_timing, "_torch_modules", lambda: (FakeTorch, object()))

    def fake_budget(fn, warmup_ms, repetition_ms):
        events.append(("budget", warmup_ms, repetition_ms))
        return 3, 7

    def fake_profile(fn, warmup_count, active_count, clear_cache, keep_profile):
        events.append(
            ("profile", warmup_count, active_count, clear_cache, keep_profile)
        )
        return 250.0

    monkeypatch.setattr(ascend_timing, "_budget_to_counts", fake_budget)
    monkeypatch.setattr(ascend_timing, "_profile_latency_us", fake_profile)

    latency = ascend_timing.measure_latency(
        lambda: events.append("fn"),
        warmup_ms=1000,
        repetition_ms=100,
    )

    assert latency == 0.25
    assert events == [
        "fn",
        "synchronize",
        ("budget", 1000, 100),
        ("profile", 3, 7, True, False),
    ]


def test_measure_latency_has_no_invalid_timing_fallback(monkeypatch):
    class FakeNpu:
        @staticmethod
        def synchronize():
            pass

    class FakeTorch:
        npu = FakeNpu()

    monkeypatch.setattr(ascend_timing, "_torch_modules", lambda: (FakeTorch, object()))
    monkeypatch.setattr(ascend_timing, "_budget_to_counts", lambda *args, **kwargs: (1, 1))
    monkeypatch.setattr(
        ascend_timing,
        "_profile_latency_us",
        lambda *args, **kwargs: float("inf"),
    )

    with pytest.raises(RuntimeError, match="invalid timing"):
        ascend_timing.measure_latency(lambda: None, 10, 10)


def test_benchmark_routes_ascend_kernel_mode_to_strict_profiler(monkeypatch):
    captured = {}

    def fake_measure(fn, warmup_ms, repetition_ms, clear_l2_cache):
        captured.update(
            fn=fn,
            warmup_ms=warmup_ms,
            repetition_ms=repetition_ms,
            clear_l2_cache=clear_l2_cache,
        )
        return 0.125

    monkeypatch.setattr(base, "vendor_name", "ascend")
    monkeypatch.setattr(base.Config, "mode", consts.BenchMode.KERNEL)
    monkeypatch.setattr(base.Config, "warm_up", 123)
    monkeypatch.setattr(base.Config, "repetition", 45)
    monkeypatch.setattr(ascend_timing, "measure_latency", fake_measure)

    benchmark = object.__new__(base.Benchmark)
    benchmark.is_backward = False
    latency = benchmark.get_latency(lambda value: value + 1, 2)

    assert latency == 0.125
    assert captured["fn"]() == 3
    assert captured["warmup_ms"] == 123
    assert captured["repetition_ms"] == 45
    assert captured["clear_l2_cache"] is True
