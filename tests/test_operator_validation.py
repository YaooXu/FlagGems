"""Regression checks for assertion and stateful measurement boundaries."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize(
    "module_path",
    [
        "benchmark/test_col_indices.py",
        "tests/test__choose_qparams_per_tensor.py",
        "tests/test__coalesce.py",
        "tests/test__dimI.py",
        "tests/test__dimV.py",
        "tests/test__dim_arange.py",
        "tests/test__efficientzerotensor.py",
        "tests/test__fw_primal.py",
        "tests/test__has_same_storage_numel.py",
        "tests/test__indices.py",
        "tests/test__make_dual.py",
        "tests/test__make_per_tensor_quantized_tensor.py",
        "tests/test__neg_view.py",
        "tests/test__neg_view_copy.py",
        "tests/test__nested_tensor_size.py",
        "tests/test__nested_tensor_storage_offsets.py",
        "tests/test__nested_tensor_strides.py",
        "tests/test__nnz.py",
        "tests/test__shape_as_tensor.py",
        "tests/test__unpack_dual.py",
        "tests/test__values.py",
        "tests/test__version.py",
        "tests/test_atleast_1d.py",
        "tests/test_atleast_2d.py",
        "tests/test_atleast_3d.py",
        "tests/test_cartesian_prod.py",
        "tests/test_ccol_indices.py",
        "tests/test_ccol_indices_copy.py",
        "tests/test_chain_matmul.py",
        "tests/test_coalesce.py",
        "tests/test_col_indices.py",
        "tests/test_col_indices_copy.py",
        "tests/test_combinations.py",
        "tests/test_copy_sparse_to_sparse_.py",
        "tests/test_crow_indices.py",
        "tests/test_crow_indices_copy.py",
        "tests/test_data.py",
        "tests/test_dense_dim.py",
        "tests/test_diagflat.py",
        "tests/test_dim.py",
        "tests/test_dstack.py",
        "tests/test_flatten_dense_tensors.py",
        "tests/test_slow_conv_transpose3d.py",
        "tests/test_sparse_bsc_tensor.py",
        "tests/test_sparse_bsr_tensor.py",
        "tests/test_sparse_coo_tensor.py",
        "tests/test_sparse_dim.py",
        "tests/test_sparse_mask.py",
        "tests/test_sparse_resize_.py",
        "tests/test_sparse_resize_and_clear_.py",
        "tests/test_adjoint.py",
    ],
)
def test_operator_collection_does_not_probe_runtime(module_path, monkeypatch):
    calls = []

    def reject_probe(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Runtime probe during test collection")

    path = Path(__file__).resolve().parents[1] / module_path
    operator = path.stem.removeprefix("test_")
    # Parameter generation may use CPU randperm for index lists; operator
    # execution and construction of probe inputs must wait until the test runs.
    for module, name in [
        (torch.ops.aten, operator),
        (tu, "make_input"),
        (torch.testing, "make_tensor"),
        (torch, "tensor"),
        (torch, "zeros"),
        (torch, "ones"),
        (torch, "empty"),
        (torch, "full"),
    ]:
        monkeypatch.setattr(module, name, reject_probe)
    runpy.run_path(str(path), run_name=f"{path.parent.name}._collection_check")
    # A probe that catches the injected error must still fail this check.
    assert not calls


def test_combinations_missing_candidate_cannot_pass_against_reference(monkeypatch):
    from . import test_combinations as cases

    missing = Mock(side_effect=LookupError("candidate missing"))
    monkeypatch.setattr(cases, "_resolve_gems_op", missing)
    with pytest.raises(LookupError, match="candidate missing"):
        cases.test_combinations_spec_shapes_value_ranges(
            (4,), ["0", "1"], torch.float32
        )
    missing.assert_called_once()


def test_version_input_error_does_not_retry_with_another_range(monkeypatch):
    from . import test__version as cases

    failed = Mock(side_effect=RuntimeError("input construction failed"))
    monkeypatch.setattr(torch.testing, "make_tensor", failed)
    with pytest.raises(RuntimeError, match="input construction failed"):
        cases._make_value_tensor(torch.int32, (4,), ["0", "1"], "cpu")
    failed.assert_called_once()


def test_sparse_copy_input_error_does_not_use_another_generator(monkeypatch):
    from . import test_copy_sparse_to_sparse_ as cases

    failed = Mock(side_effect=RuntimeError("input construction failed"))
    monkeypatch.setattr(tu, "make_input", failed)
    with pytest.raises(RuntimeError, match="input construction failed"):
        cases._make_values((4,), torch.uint8, value_range=["0", "1"])
    failed.assert_called_once()


@pytest.mark.parametrize(
    "operator", ["copy_sparse_to_sparse_", "sparse_mask", "sparse_resize_"]
)
def test_sparse_storage_operations_reject_small_value_changes(operator):
    from . import test_copy_sparse_to_sparse_ as copy_cases
    from . import test_sparse_mask as mask_cases
    from . import test_sparse_resize_ as resize_cases

    def corrupted(*args, **kwargs):
        result = getattr(torch.ops.aten, operator)(*args, **kwargs)
        result._values().add_(1e-6)
        return result

    checks = {
        "copy_sparse_to_sparse_": lambda: copy_cases.test_copy_sparse_to_sparse_(
            ((4, 5), 2, 3), torch.float32, False
        ),
        "sparse_mask": lambda: mask_cases.test_sparse_mask_value_ranges(
            (4, 5), ["0", "1"], torch.float32
        ),
        "sparse_resize_": lambda: resize_cases.test_sparse_resize_(
            ((4, 5), 2, 3, (6, 5), 2, 0), torch.float32
        ),
    }
    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            checks[operator]()


def test_sparse_copy_rejects_changed_coalesced_flag():
    from . import test_copy_sparse_to_sparse_ as cases

    def corrupted(dst, src, non_blocking):
        result = torch.ops.aten.copy_sparse_to_sparse_(dst, src, non_blocking)
        result._coalesced_(not result.is_coalesced())
        return result

    with testing.override_gems_op("copy_sparse_to_sparse_", corrupted):
        with pytest.raises(AssertionError):
            cases.test_copy_sparse_to_sparse_(((4, 5), 2, 3), torch.float32, False)


@pytest.mark.parametrize("layout", ["coo", "csc", "bsc"])
def test_sparse_constructors_reject_small_storage_changes(layout):
    from . import test_sparse_bsc_tensor as bsc
    from . import test_sparse_coo_tensor as coo
    from . import test_sparse_csc_tensor as csc

    operator = f"sparse_{layout}_tensor"

    def corrupted(*args, **kwargs):
        result = getattr(torch.ops.aten, operator)(*args, **kwargs).clone()
        values = result._values() if layout == "coo" else result.values()
        values.add_(1e-6)
        return result

    checks = {
        "coo": lambda: coo.test_sparse_coo_tensor_indices_size(
            coo._COO_2D_CASES[0], torch.float32
        ),
        "csc": lambda: csc.test_sparse_csc_tensor(
            (4, 4), 4, torch.float32, torch.int64, ["0", "1"]
        ),
        "bsc": lambda: bsc.test_sparse_bsc_tensor(
            bsc._BSC_CASES[0], torch.float32, torch.int64
        ),
    }
    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            checks[layout]()


@pytest.mark.parametrize("component", ["values", "rows"])
def test_csc_candidate_cannot_change_reference_through_shared_inputs(component):
    from . import test_sparse_csc_tensor as cases

    def corrupted(ccol, row, values, *args, **kwargs):
        if component == "values":
            values.add_(1)
        else:
            row.copy_((row + 1) % 4)
        return torch.ops.aten.sparse_csc_tensor(ccol, row, values, *args, **kwargs)

    with testing.override_gems_op("sparse_csc_tensor", corrupted):
        with pytest.raises(AssertionError):
            cases.test_sparse_csc_tensor(
                (4, 4), 4, torch.float32, torch.int64, ["0", "1"]
            )


@pytest.mark.parametrize("value", [0.5, False])
def test_version_rejects_nonintegral_tensor_results(value):
    from . import test__version as cases

    def invalid(inp):
        return torch.tensor(value, device=inp.device)

    with testing.override_gems_op("_version", invalid):
        with pytest.raises(AssertionError):
            cases.test__version_fresh((4,), torch.float32)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_version_accepts_integral_tensor_results(dtype):
    from . import test__version as cases

    def valid(inp):
        return torch.tensor(
            torch.ops.aten._version(inp), dtype=dtype, device=inp.device
        )

    with testing.override_gems_op("_version", valid):
        cases.test__version_fresh((4,), torch.float32)


def test_quantization_params_reject_boolean_zero_point():
    from . import test__choose_qparams_per_tensor as cases

    def invalid(inp, reduce_range):
        scale, zero_point = torch.ops.aten._choose_qparams_per_tensor(inp, reduce_range)
        return scale, bool(zero_point)

    with testing.override_gems_op("_choose_qparams_per_tensor", invalid):
        with pytest.raises(AssertionError):
            cases.test__choose_qparams_per_tensor_constant(0.0, torch.float32, False)


@pytest.mark.parametrize(
    "operator,args",
    [
        ("combinations", (8, 2, False, torch.float32)),
        ("diagflat", ((3, 4), 1, torch.float32)),
        ("dstack", ([(2, 3), (2, 3)], torch.float32)),
        ("flatten_dense_tensors", ([(2, 3), (2, 3)], torch.float32)),
        ("_fw_primal", ((3, 4), torch.float32)),
        ("_remove_batch_dim", ((1, 3), 0, 2, torch.float32)),
    ],
)
def test_gather_backward_cases_reject_small_forward_errors(operator, args):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    def corrupted(*args, **kwargs):
        return getattr(torch.ops.aten, operator)(*args, **kwargs) + 1e-6

    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_backward")(*args)


def test_diagflat_rejects_small_gradient_errors():
    from . import test_diagflat as cases

    class CorruptedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, offset):
            ctx.shape = inp.shape
            ctx.offset = offset
            return torch.ops.aten.diagflat(inp, offset)

        @staticmethod
        def backward(ctx, grad):
            return torch.ops.aten.diag(grad, ctx.offset).reshape(ctx.shape) + 1e-6, None

    with testing.override_gems_op("diagflat", CorruptedGradient.apply):
        with pytest.raises(AssertionError):
            cases.test_diagflat_backward((3, 4), 1, torch.float32)


@pytest.mark.parametrize(
    "operator,case_name,args",
    [
        ("_fw_primal", "test__fw_primal_rejects_non_tensor", ()),
        (
            "_slow_conv2d_backward",
            "test__slow_conv2d_backward_negative_non_4d_grad_output",
            (),
        ),
        (
            "slow_conv_dilated2d",
            "test_slow_conv_dilated2d_rejects_unsupported_dtype",
            (torch.int32,),
        ),
    ],
)
def test_negative_cases_require_a_candidate(monkeypatch, operator, case_name, args):
    cases = importlib.import_module(f".test_{operator}", package=__package__)
    missing = Mock(side_effect=LookupError("candidate missing"))
    monkeypatch.setattr(cases, "_resolve_gems_op", missing)
    with pytest.raises(LookupError, match="candidate missing"):
        getattr(cases, case_name)(*args)
    missing.assert_called_once()


@pytest.mark.parametrize(
    "operator,shape", [("atleast_1d", ()), ("atleast_2d", (3,)), ("atleast_3d", (2, 3))]
)
def test_atleast_backward_rejects_constant_gradient(operator, shape):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    class ConstantGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            ctx.shape = inp.shape
            return getattr(torch.ops.aten, operator)(inp)

        @staticmethod
        def backward(ctx, grad):
            return torch.ones_like(grad).reshape(ctx.shape)

    with testing.override_gems_op(operator, ConstantGradient.apply):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_backward")(shape, torch.float32)


def test_detach_copy_rejects_an_implemented_backward():
    from . import test_detach_copy as cases

    with testing.override_gems_op("detach_copy", lambda inp: inp.clone()):
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            cases.test_detach_copy_no_backward((3, 4), torch.float32)


@pytest.mark.parametrize("operator", ["_nested_tensor_size", "_nested_tensor_strides"])
def test_nested_metadata_stays_on_cpu_with_cpu_reference(operator, monkeypatch):
    cases = importlib.import_module(f".test_{operator}", package=__package__)
    monkeypatch.setattr(utils, "TO_CPU", True)

    def misplaced(inp):
        return getattr(torch.ops.aten, operator)(inp).to(inp.device)

    with testing.override_gems_op(operator, misplaced):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_nan_inf_values")(torch.float32, "nan")


@pytest.mark.parametrize(
    "operator",
    [
        "ccol_indices",
        "crow_indices",
        "col_indices",
        "ccol_indices_copy",
        "crow_indices_copy",
        "col_indices_copy",
    ],
)
def test_compressed_indices_reject_widened_index_dtype(operator):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    def widened(inp):
        return getattr(torch.ops.aten, operator)(inp).to(torch.int64)

    with testing.override_gems_op(operator, widened):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_index_layouts")(
                cases._INDEX_LAYOUT_CASES[0], (), (), torch.float32, torch.int32
            )


@pytest.mark.parametrize(
    "n,r,with_replacement", [(8, 1, False), (8, 1, True), (1, 2, False)]
)
def test_combinations_nonreducing_backward_rejects_small_errors(n, r, with_replacement):
    from . import test_combinations as cases

    class CorruptedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, r, replacement):
            ctx.shape = inp.shape
            return torch.ops.aten.combinations(inp, r, replacement)

        @staticmethod
        def backward(ctx, grad):
            if grad.numel() == 0:
                return (
                    torch.full(ctx.shape, 1e-6, dtype=grad.dtype, device=grad.device),
                    None,
                    None,
                )
            return grad.reshape(ctx.shape) + 1e-6, None, None

    with testing.override_gems_op("combinations", CorruptedGradient.apply):
        with pytest.raises(AssertionError):
            cases.test_combinations_backward(n, r, with_replacement, torch.float32)


def test_combinations_zero_r_rejects_a_gradient_connection():
    from . import test_combinations as cases

    with testing.override_gems_op("combinations", lambda inp, r, replacement: inp[:0]):
        with pytest.raises(AssertionError):
            cases.test_combinations_zero_r_no_autograd(8, False, torch.float32)
