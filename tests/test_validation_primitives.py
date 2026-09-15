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
    testing.assert_equal(source, source.clone(), equal_nan=True)
    testing.assert_close(source, source.clone(), dtype, equal_nan=True)
    with pytest.raises(AssertionError):
        testing.assert_equal(source.float(), source, equal_nan=True)
    bad = torch.tensor([0.0, 2.0, float("nan")]).to(dtype)
    with pytest.raises(AssertionError):
        testing.assert_equal(bad, source, equal_nan=True)


def test_exact_copy_comparison_rejects_small_error():
    source = torch.ones(10)
    with pytest.raises(AssertionError):
        tu.assert_result_equal(source + 1e-4, source)
    testing.assert_equal(3, 3)
    with pytest.raises(AssertionError):
        testing.assert_equal(3, 4)


def test_snapshot_preserves_layout_aliases_and_independent_gradients():
    base = torch.arange(40.0).reshape(5, 8).requires_grad_()
    first, second = base[1:, ::2], base[:, 2:]
    a, b = testing.clone_inputs((first, second))
    assert a.stride() == first.stride()
    assert a.storage_offset() == first.storage_offset()
    assert torch._C._is_alias_of(a, b)
    assert not torch._C._is_alias_of(a, base)
    assert a.is_leaf and a.requires_grad
    with torch.no_grad():
        a[0, 1] = -99
    assert b[1, 0] == -99
    assert first[0, 1] != -99
    a.sum().backward()
    assert base.grad is None


def test_reference_is_independent_even_when_no_upcast_or_cpu_copy(monkeypatch):
    monkeypatch.setattr(utils, "TO_CPU", False)
    source = torch.arange(6.0)
    reference = utils.to_reference(source, independent=True)
    source.add_(100)
    testing.assert_equal(reference, torch.arange(6.0))


def test_checked_candidate_rejects_input_mutation():
    def corrupt(value):
        value.add_(100)
        return value.view(-1)

    with testing.override_gems_op("atleast_1d", corrupt):
        with pytest.raises(AssertionError):
            tu.resolve_gems_op("atleast_1d")(torch.arange(5.0))


def test_checked_candidate_allows_real_out_alias():
    value = torch.ones(4)
    with testing.override_gems_op("add", torch.ops.aten.add):
        result = tu.resolve_gems_op("add")(value, 2, out=value)
    assert result is value
    testing.assert_equal(value, torch.full((4,), 3.0))


def test_checked_inplace_candidate_protects_other_inputs():
    def corrupt(target, source):
        source.zero_()
        return target.copy_(source)

    with testing.override_gems_op("copy_", corrupt):
        with pytest.raises(AssertionError):
            tu.resolve_gems_op("copy_")(torch.zeros(4), torch.ones(4))


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
        "sparse_resize_and_clear_", clear, is_inplace=True, gems_op=clear
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
        "validation_", mutate, is_inplace=True, gems_op=mutate
    )
    original = testing.clone_inputs

    def snapshot(x, *args, **kwargs):
        events.append(("prepare", None))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(testing, "clone_inputs", snapshot)
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


@pytest.mark.parametrize("dtype", [torch.bool, torch.float8_e4m3fn, torch.float8_e5m2])
def test_snapshot_checks_lazy_negative_inputs_without_eager_neg(dtype):
    value = torch.tensor([0.0, 1.0]).to(dtype)
    lazy = torch._neg_view(value)
    with testing.override_gems_op("_neg_view", torch.ops.aten._neg_view):
        result = tu.resolve_gems_op("_neg_view")(lazy)
    assert not result.is_neg()
    testing.assert_equal(result, value)


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
    original = testing.clone_inputs

    def snapshot(value, *args, **kwargs):
        phase.append("prepare")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(testing, "clone_inputs", snapshot)

    def mutate(x):
        phase.append("call")
        x.add_(1)

    bench = bench_config.Benchmark(
        "validation_", mutate, gems_op=mutate, is_inplace=True
    )
    result = bench.get_latency(mutate, torch.zeros(1))
    assert result == pytest.approx(1.0)
    start = phase.index("clock")
    assert phase[start:] == ["clock", "call", "clock"]


def test_native_metadata_expectation_still_protects_input_values():
    source = torch.ones(2, 3)
    reference = source.clone()
    reference.unsqueeze_(0)

    def metadata_only(x):
        return x.unsqueeze_(0)

    with testing.override_gems_op("metadata_change", metadata_only):
        result = tu.resolve_gems_op(
            "metadata_change", expected_input_metadata={0: reference}
        )(source)
        assert result.shape == (1, 2, 3)

    def corrupt(x):
        x.add_(1)
        return x.unsqueeze_(0)

    with testing.override_gems_op("metadata_change", corrupt):
        with pytest.raises(AssertionError):
            tu.resolve_gems_op(
                "metadata_change", expected_input_metadata={0: reference}
            )(torch.ones(2, 3))


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
