"""Regression checks for assertion and stateful measurement boundaries."""

from types import SimpleNamespace

import pytest
import torch

import flag_gems
from flag_gems import testing

from . import accuracy_utils as utils
from . import test_utils as tu


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_comparison_preserves_dtype_and_rejects_wrong_values(dtype):
    source = torch.tensor([0.0, 1.0, float("nan")]).to(dtype)
    tu.assert_result_equal(source, source.clone())
    tu.assert_result_close(source, source.clone())
    with pytest.raises(AssertionError):
        tu.assert_result_equal(source.float(), source)
    bad = torch.tensor([0.0, 2.0, float("nan")]).to(dtype)
    with pytest.raises(AssertionError):
        tu.assert_result_equal(bad, source)


def test_exact_copy_comparison_rejects_small_error():
    source = torch.ones(10)
    with pytest.raises(AssertionError):
        tu.assert_result_equal(source + 1e-4, source)
    testing.assert_equal(3, 3)
    with pytest.raises(AssertionError):
        testing.assert_equal(3, 4)


def test_reference_is_independent_even_when_no_upcast_or_cpu_copy(monkeypatch):
    monkeypatch.setattr(utils, "TO_CPU", False)
    source = torch.arange(6.0)
    reference = tu.to_reference(source)
    source.add_(100)
    testing.assert_equal(reference, torch.arange(6.0))


def test_dtype_probe_rejects_invalid_signature():
    with pytest.raises(RuntimeError, match="Inconclusive dtype probe"):
        tu.supported_dtypes("add", [torch.float32])


def test_dtype_probe_input_failure_is_not_unsupported(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("input construction failed")

    monkeypatch.setattr(torch.testing, "make_tensor", broken)
    with pytest.raises(RuntimeError, match="input construction failed"):
        tu.supported_dtypes("atleast_1d", [torch.float32])


@pytest.mark.parametrize(
    "dtype,scenario",
    tu.special_value_cases([torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16]),
)
def test_special_scenarios_retain_representable_payload(dtype, scenario):
    x = tu.make_special_input(dtype, scenario).float()
    assert torch.isnan(x).any().item() == (scenario in ("nan", "mixed"))
    assert torch.isinf(x).any().item() == (scenario in ("inf", "mixed"))


@pytest.fixture
def bench_config(monkeypatch):
    from benchmark import base
    from benchmark.consts import BenchMode

    config = SimpleNamespace(mode=BenchMode.OPERATOR, warm_up=0.03, repetition=0.03)
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(synchronize=lambda: None)
    )
    return base


def test_missing_direct_candidate_never_falls_back(bench_config):
    bench = bench_config.Benchmark("missing_for_validation", lambda x: x, gems_op=None)
    with pytest.raises(LookupError):
        bench._candidate_context_and_op()


def test_explicit_none_resolves_existing_public_candidate(bench_config, monkeypatch):
    def op(x):
        return x

    monkeypatch.setattr(flag_gems, "validation_op", op, raising=False)
    bench = bench_config.Benchmark("validation_op", lambda x: x, gems_op=None)
    _, resolved = bench._candidate_context_and_op()
    assert resolved is op


def test_stateful_latency_restores_before_every_call_and_each_side(bench_config):
    seen = []
    source = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.ones(2), (2, 2)
    )

    def clear(x):
        seen.append((x._nnz(), tuple(x.shape)))
        x.sparse_resize_and_clear_((4, 4), 2, 0)

    bench = bench_config.Benchmark(
        "sparse_resize_and_clear_",
        clear,
        is_inplace=True,
        fresh_inputs=True,
        gems_op=clear,
    )
    assert bench.get_latency(clear, source) > 0
    assert bench.get_latency(clear, source) > 0
    assert len(seen) >= 4
    assert set(seen) == {(2, (2, 2))}
    assert source._nnz() == 2


def test_stateful_profile_restores_outside_each_capture(bench_config, monkeypatch):
    events = []

    def mutate(x):
        events.append(("call", x.item()))
        x.add_(1)

    bench = bench_config.Benchmark(
        "validation_", mutate, is_inplace=True, fresh_inputs=True, gems_op=mutate
    )
    original = bench_config._clone_benchmark_inputs

    def snapshot(x, *args, **kwargs):
        events.append(("prepare", None))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(bench_config, "_clone_benchmark_inputs", snapshot)
    monkeypatch.setattr(
        bench, "_external_profiler_start", lambda: events.append(("start", None))
    )
    monkeypatch.setattr(
        bench, "_external_profiler_stop", lambda: events.append(("stop", None))
    )
    bench._run_candidate_input(
        (torch.zeros(1),), (torch.zeros(1),), warmup=2, iterations=3, profile=True
    )
    assert [v for k, v in events if k == "call"] == [0.0] * 5
    active = False
    for name, _ in events:
        if name == "start":
            active = True
        elif name == "stop":
            active = False
        elif name == "prepare":
            assert not active


def test_stateful_measurement_boundary_excludes_restoration(bench_config, monkeypatch):
    phase = []
    ticks = iter([0.0, 0.001])
    bench_config.Config.warm_up = 0
    bench_config.Config.repetition = 0
    monkeypatch.setattr(
        bench_config.time,
        "perf_counter",
        lambda: (phase.append("clock"), next(ticks))[1],
    )
    original = bench_config._clone_benchmark_inputs

    def snapshot(value, *args, **kwargs):
        phase.append("prepare")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(bench_config, "_clone_benchmark_inputs", snapshot)

    def mutate(x):
        phase.append("call")
        x.add_(1)

    bench = bench_config.Benchmark(
        "validation_", mutate, gems_op=mutate, is_inplace=True, fresh_inputs=True
    )
    result = bench.get_latency(mutate, torch.zeros(1))
    assert result == pytest.approx(1.0)
    start = phase.index("clock")
    assert phase[start:] == ["clock", "call", "clock"]


def test_arithmetic_helper_rejects_candidate_dtype_change():
    with pytest.raises(AssertionError):
        tu.assert_result_close(
            torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float32)
        )


@pytest.mark.parametrize("unsupported", ["cudagraph", "backward"])
def test_stateful_unsupported_measurement_never_reports_forward_latency(
    bench_config, unsupported
):
    from benchmark.consts import BenchMode

    if unsupported == "cudagraph":
        bench_config.Config.mode = BenchMode.CUDAGRAPH
    bench = bench_config.Benchmark(
        "validation_",
        lambda x: x,
        is_inplace=True,
        fresh_inputs=True,
        is_backward=unsupported == "backward",
    )
    with pytest.raises(ValueError):
        bench.get_latency(lambda x: x, torch.ones(1))


@pytest.mark.parametrize(
    "complex_dtype,real_dtype",
    [
        (torch.complex32, torch.float16),
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
    ],
)
def test_complex_bounds_follow_component_dtype(complex_dtype, real_dtype):
    assert tu.dtype_bounds(complex_dtype) == (
        torch.finfo(real_dtype).min,
        torch.finfo(real_dtype).max,
    )


def test_existing_inplace_benchmark_keeps_legacy_timing(bench_config, monkeypatch):
    from benchmark import base

    bench = base.Benchmark("add_", lambda x: x, is_inplace=True)
    assert not bench.fresh_inputs
    monkeypatch.setattr(base.Config, "mode", base.consts.BenchMode.OPERATOR)
    monkeypatch.setattr(base, "get_iter_count", lambda fn: (1, 1))
    calls = []
    bench.get_latency(lambda x: calls.append(x), torch.ones(1))
    assert len(calls) == 2
    assert calls[0] is calls[1]


def test_reference_retains_view_metadata_and_independent_gradients():
    source = torch.arange(40.0).reshape(5, 8)[1:, ::2].requires_grad_()
    reference = tu.to_reference(source)
    assert reference.stride() == source.stride()
    assert reference.storage_offset() == source.storage_offset()
    assert not torch._C._is_alias_of(reference, source)
    with torch.no_grad():
        source.add_(100)
    torch.testing.assert_close(reference, torch.arange(40.0).reshape(5, 8)[1:, ::2])
    reference.sum().backward()
    assert source.grad is None
