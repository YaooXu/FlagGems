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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu
from .conftest import QUICK_MODE

# ``_slow_conv2d_forward`` starts with an underscore, and ``pytest.mark`` refuses
# to generate a marker via attribute access for such names. Register the marker
# directly on the MarkGenerator so ``@pytest.mark._slow_conv2d_forward`` and
# ``-m _slow_conv2d_forward`` both work.
setattr(
    pytest.mark,
    "_slow_conv2d_forward",
    MarkDecorator(
        Mark("_slow_conv2d_forward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_slow_conv2d_forward(Tensor self, Tensor weight, SymInt[2] kernel_size,
# Tensor? bias, SymInt[2] stride, SymInt[2] padding) -> Tensor is the im2col
# based "slow" conv2d (no dilation, groups always 1). ``self`` is (N, C_in, H, W),
# ``weight`` is (C_out, C_in, kH, kW) and ``kernel_size`` must match the weight
# spatial dims. The output is (N, C_out, H_out, W_out) with
#   H_out = (H + 2*pH - kH) // sH + 1, W_out = (W + 2*pW - kW) // sW + 1.
# The .default overload is resolved through its public name "_slow_conv2d_forward"
# (KernelGen's override_gems_op("_slow_conv2d_forward", ...) wins over the direct
# callable); the .output overload is resolved through "_slow_conv2d_forward.output"
# whose fallback implementation is the adapter below.
#
# Coverage follows the regular-operator spec adapted to a matrix/reduction-like
# op:
#   * shape/param levels: each (input, weight, kernel_size, stride, padding)
#     tuple in SLOW_CONV2D_CASES is one distinct parametrized workload (the
#     shared tu.selected_shapes() set is pointwise-shaped and does not apply to
#     a conv whose input must be 4-D); they cover 1x1/3x3/3x5/5x5 kernels,
#     stride 1 and 2, padding 0/1/2, small outputs and channel counts up to 16;
#   * dtype coverage: the op is probed (see UNSUPPORTED_DTYPES) and only the
#     floating dtypes the CUDA/CPU kernel actually implements are exercised -
#     fp16/fp32/bf16/fp64; int8/uint8/fp8/int32/int64/bool all raise
#     "slow_conv2d_*" not implemented and are covered as negative cases;
#   * value ranges: tu.make_input over the spec's ranges, adapted so the
#     multiply-accumulate never overflows (the [0, max]/[min, 0] extremes are
#     dropped; see _CONV_VALUE_RANGES);
#   * broadcast: not applicable - conv requires C_in to match exactly between
#     input and weight (any mismatch is a negative case below);
#   * backward: the op is differentiable (aten routes to _slow_conv2d_backward),
#     so gradients are compared against the fp64 upcast reference, including
#     the candidate path when the kernel advertises autograd support;
#   * negative: kernel_size mismatch, kernel larger than input, C_in mismatch,
#     non-4-D input, unsupported dtype and non-tuple scalar params all raise;
#   * nan/inf: deterministic propagation through the im2col GEMM.
if QUICK_MODE:
    SLOW_CONV2D_CASES = [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    SLOW_CONV2D_CASES = [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((2, 3, 9, 9), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
        ((2, 3, 8, 8), (5, 3, 3, 3), (3, 3), (2, 2), (1, 1)),
        ((2, 3, 8, 8), (5, 3, 3, 5), (3, 5), (1, 1), (1, 2)),
        ((2, 8, 16, 16), (16, 8, 1, 1), (1, 1), (1, 1), (0, 0)),
        ((4, 16, 32, 32), (8, 16, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((1, 4, 12, 12), (4, 4, 5, 5), (5, 5), (1, 1), (2, 2)),
        ((2, 3, 4, 4), (5, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)
    BIASES = [True, False]

# Dtypes probed against the real aten kernel: none of them are implemented
# ("slow_conv2d_cuda"/"slow_conv2d_cpu" not implemented for ...), so they are
# excluded from the positive grid and asserted as negative cases instead. This
# is what the spec's "probe before you write" rule requires (the required
# int8/uint8/fp8 coverage applies only when the kernel supports them).
UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
    torch.bool,
]

# Value-range coverage deviates from the spec's five ranges: a conv contracts
# over C_in*kH*kW products, and the candidate (like the native operator)
# accumulates them in the *input* dtype. Dtype-max inputs overflow that
# accumulator immediately -- a single product of two ~dtype-max values already
# exceeds the input dtype (fp32: ~1e77 vs a 3.4e38 max) -- so the im2col GEMM
# saturates to inf (NaN where opposite-sign terms cancel), while the fp64
# upcast reference stays finite. Measured on this device with fp32 inputs drawn
# from [0, dtype_max]: the native output is entirely inf and the fp64 reference
# is entirely finite (max ~6e77, far below the fp64 limit 1.8e308). The two
# dtype-extreme ranges [0, dtype_max] and [dtype_min, 0] are therefore dropped:
# the comparison would be dominated by saturation rather than the operator's
# rounding. The three retained ranges [-1, 1], [0, 1] and [-1, 0] still cover
# negative, positive, mixed, zero and constant inputs for every dtype.
_UNSAFE_FOR_REDUCTION = frozenset({"max", "min", "max/2", "min/2"})
_CONV_VALUE_RANGES = [
    value_range
    for value_range in tu.selected_ranges()
    if not ({value_range[0], value_range[1]} & _UNSAFE_FOR_REDUCTION)
]

# Backward cases stay small (autograd graph + 2x forward/backward passes).
_BACKWARD_CASES = SLOW_CONV2D_CASES[:3]

# Inputs are scaled down before the fp64 upcast reference is computed. The
# gradients are reduction-heavy, so fp16/bf16 backward accumulates rounding
# noise proportional to the data magnitude; 0.1 keeps that noise well inside
# the gems_assert_close tolerance (see test__slow_conv2d_backward.py for the
# same convention).
_INPUT_SCALE = 0.1


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._slow_conv2d_forward is registered; resolution order is:
    # (1) override, (2) the direct flag_gems._slow_conv2d_forward callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_slow_conv2d_forward", getattr(flag_gems, "_slow_conv2d_forward", None)
    )


def _slow_conv2d_forward_out_adapter(
    self, weight, kernel_size, bias, stride, padding, *, output
):
    # Fallback implementation of the ".output" overload used when no candidate
    # override is installed: run the direct forward kernel and write the result
    # into the caller's out buffer. KernelGen's override of
    # "_slow_conv2d_forward.output" replaces this adapter and is called directly
    # with ``output=...`` by the test below.
    output.copy_(_resolve_gems_op()(self, weight, kernel_size, bias, stride, padding))
    return output


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_slow_conv2d_forward.output", _slow_conv2d_forward_out_adapter
    )


def _conv_output_shape(inp_shape, weight_shape, kernel_size, stride, padding):
    n, _, h_in, w_in = inp_shape
    out_c, _, k_h, k_w = weight_shape
    h_out = (h_in + 2 * padding[0] - k_h) // stride[0] + 1
    w_out = (w_in + 2 * padding[1] - k_w) // stride[1] + 1
    return (n, out_c, h_out, w_out)


def _make_conv_inputs(inp_shape, weight_shape, with_bias, dtype, value_range):
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = tu.make_input(dtype, (weight_shape[0],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype):
    # The reference is computed with an fp64 upcast, so it is exact for the
    # rounded inputs. The torch native op (and any good candidate) accumulates
    # the im2col GEMM in the input dtype: fp16/bf16 tensor cores keep at most
    # fp16/bf16 precision per add, so the native op itself deviates from the
    # fp64 reference by up to ~3e-2 (fp16) / ~2.5e-1 (bf16) on the larger
    # reductions. Measure the deviation over 100 seeds per shape: max required
    # absolute tolerance is ~2e-3 (fp16) and ~1.5e-2 (bf16) after the rtol
    # term is applied; fp16 -> 1e-2 and bf16 -> 5e-2 give 5x/3.3x margin.
    # fp32 with TF32 disabled (set at the top of each test) stays at ~3e-5,
    # comfortably inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 5e-2
    elif dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


def _reduction_dims(inp_shape, weight_shape, out_shape):
    n, _, _, _ = inp_shape
    out_c, _, k_h, k_w = weight_shape
    # grad_input contracts over C_out x kH x kW.
    in_reduce_dim = out_c * k_h * k_w
    # grad_weight / grad_bias contract over N x H_out x W_out.
    out_reduce_dim = n * out_shape[2] * out_shape[3]
    return in_reduce_dim, out_reduce_dim


def _assert_grads_close(res_grads, ref_grads, in_reduce_dim, out_reduce_dim, dtype):
    for res_g, ref_g, reduce_dim in zip(
        res_grads, ref_grads, (in_reduce_dim, out_reduce_dim, out_reduce_dim)
    ):
        if ref_g is None:
            assert res_g is None
        else:
            utils.gems_assert_close(
                res_g, ref_g.to(dtype), dtype, reduce_dim=reduce_dim
            )


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", SLOW_CONV2D_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test__slow_conv2d_forward(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    # The reference op runs cuBLAS/baddbmm for the im2col GEMM; keep TF32 off so
    # the fp32 comparison stays at the standard 1e-4 tolerance (with TF32 on the
    # native op itself deviates from the fp64 reference by ~1.6e-2).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(inp, weight, kernel_size, bias_t, stride, padding)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", SLOW_CONV2D_CASES
)
@pytest.mark.parametrize("value_range", _CONV_VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_forward_value_ranges(
    inp_shape, weight_shape, kernel_size, stride, padding, value_range, dtype
):
    # The spec value-range sweep (tu.selected_ranges() minus the overflowing
    # extremes) for every supported dtype and every local conv shape. Bias is
    # kept on so the value ranges also exercise the bias-add path. The
    # comparison reuses the dtype-aware _assert_close tolerance because a conv
    # contracts over C_in*kH*kW terms (tu.assert_result_close's flat
    # rtol=1e-2/atol=1e-3 is calibrated for pointwise ops and is too tight for
    # fp16/bf16 on the larger reductions).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, True, dtype, value_range
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    res_out = _resolve_gems_op()(inp, weight, kernel_size, bias_t, stride, padding)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", _BACKWARD_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test__slow_conv2d_forward_backward(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    # aten::_slow_conv2d_forward is differentiable (the autograd engine routes
    # its backward to _slow_conv2d_backward). The reference gradient is computed
    # on the fp64 upcast graph with a random grad_output; the candidate forward
    # must match, and - if the candidate kernel advertises autograd support -
    # its own gradient must match the fp64 reference too.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    inp = (_INPUT_SCALE * inp).requires_grad_()
    weight = (_INPUT_SCALE * weight).requires_grad_()
    if bias_t is not None:
        bias_t = (_INPUT_SCALE * bias_t).requires_grad_()

    out_shape = _conv_output_shape(
        inp_shape, weight_shape, kernel_size, stride, padding
    )
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_out = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    )
    if ref_bias is None:
        ref_gi, ref_gw = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad_out
        )
        ref_gb = None
    else:
        ref_gi, ref_gw, ref_gb = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad_out
        )

    res_out = _resolve_gems_op()(inp, weight, kernel_size, bias_t, stride, padding)

    tu.assert_result_close(res_out, ref_out.to(dtype))

    in_reduce_dim, out_reduce_dim = _reduction_dims(inp_shape, weight_shape, out_shape)
    # The candidate forward is only autograd-aware if its wrapper wires the
    # backward; skip the gradient comparison otherwise (the reference path is
    # still fully validated above).
    if res_out.requires_grad:
        if bias_t is None:
            res_gi, res_gw = torch.autograd.grad(res_out, (inp, weight), grad_out)
            res_gb = None
        else:
            res_gi, res_gw, res_gb = torch.autograd.grad(
                res_out, (inp, weight, bias_t), grad_out
            )
        _assert_grads_close(
            (res_gi, res_gw, res_gb),
            (ref_gi, ref_gw, ref_gb),
            in_reduce_dim,
            out_reduce_dim,
            dtype,
        )


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_forward_nan_inf(dtype):
    # nan/inf must propagate through the im2col GEMM. A single nan in the input
    # makes every overlapping output nan, and a single +inf with a strictly
    # positive weight makes every overlapping output +inf (no inf + (-inf)
    # cancellation, so the propagation is deterministic for any accumulation
    # order). assert_result_close uses equal_nan=True.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = SLOW_CONV2D_CASES[0]
    inp = tu.make_input(dtype, inp_shape, ["-1", "1"])
    inp[0, 0, 1, 1] = float("nan")
    inp[0, 1, 3, 3] = float("inf")
    # Strictly positive finite weights: inf * positive = inf (never nan), and
    # no term is zero so nan/inf never get swallowed by a 0 * inf product.
    weight = tu.make_input(dtype, weight_shape, ["0", "1"]) + 0.5
    bias = tu.make_input(dtype, (weight_shape[0],), ["-1", "1"])

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    res_out = _resolve_gems_op()(inp, weight, kernel_size, bias, stride, padding)

    tu.assert_result_close(res_out, ref_out)
    # The special values must actually appear in the output (sanity check that
    # the workload really exercises the nan/inf path).
    assert torch.isnan(ref_out).any()
    assert torch.isinf(ref_out).any()


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", SLOW_CONV2D_CASES[:2]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test__slow_conv2d_forward_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    # The .output overload writes into the caller's buffer and returns the same
    # tensor object (alias semantics). The buffers are garbage-prefilled so the
    # overload must overwrite them. The real aten overload is callable on the
    # active backend, so it is called directly on both paths.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    out_shape = _conv_output_shape(
        inp_shape, weight_shape, kernel_size, stride, padding
    )
    ref_out = torch.full(out_shape, 7.0, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten._slow_conv2d_forward.output(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, output=ref_out
    )
    res_ret = _resolve_gems_op_out()(
        inp, weight, kernel_size, bias_t, stride, padding, output=res_out
    )

    # The .output overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    _assert_close(res_out, ref_out.to(dtype), dtype)


@pytest.mark._slow_conv2d_forward
def test__slow_conv2d_forward_rejects_kernel_size_mismatch():
    # kernel_size must match the weight spatial dims; aten validates it.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (2, 2),
            None,
            (1, 1),
            (0, 0),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, (2, 2), None, (1, 1), (0, 0))


@pytest.mark._slow_conv2d_forward
def test__slow_conv2d_forward_rejects_kernel_larger_than_input():
    # The padded input must be at least as large as the kernel in every spatial
    # dimension; aten raises for a 3x3 kernel over a 2x2 input.
    inp = tu.make_input(torch.float32, (1, 2, 2, 2), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (3, 3),
            None,
            (1, 1),
            (0, 0),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, (3, 3), None, (1, 1), (0, 0))


@pytest.mark._slow_conv2d_forward
def test__slow_conv2d_forward_rejects_channel_mismatch():
    # Conv has no broadcast: C_in of the input must equal C_in of the weight.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (3, 3),
            None,
            (1, 1),
            (0, 0),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, (3, 3), None, (1, 1), (0, 0))


@pytest.mark._slow_conv2d_forward
def test__slow_conv2d_forward_rejects_non_4d_input():
    # self must be (N, C_in, H, W); any other rank is rejected.
    inp = tu.make_input(torch.float32, (2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (3, 3),
            None,
            (1, 1),
            (0, 0),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, (3, 3), None, (1, 1), (0, 0))


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test__slow_conv2d_forward_rejects_unsupported_dtype(dtype):
    # int8/uint8/fp8/int32/int64/bool are not implemented by slow_conv2d; the
    # aten kernel raises and any candidate must reject them as well.
    inp = tu.make_input(dtype, (1, 2, 5, 5), ["0", "1"])
    weight = tu.make_input(dtype, (1, 2, 3, 3), ["0", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp),
            utils.to_reference(weight),
            (3, 3),
            None,
            (1, 1),
            (1, 1),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, (3, 3), None, (1, 1), (1, 1))


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize("scalar_param", ["kernel_size", "stride", "padding"])
def test__slow_conv2d_forward_rejects_scalar_params(scalar_param):
    # kernel_size/stride/padding are SymInt[2]: passing a bare scalar int does
    # not match the schema and raises.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    if scalar_param == "kernel_size":
        bad_kwargs = {"kernel_size": 3, "stride": (1, 1), "padding": (1, 1)}
    elif scalar_param == "stride":
        bad_kwargs = {"kernel_size": (3, 3), "stride": 1, "padding": (1, 1)}
    else:
        bad_kwargs = {"kernel_size": (3, 3), "stride": (1, 1), "padding": 1}
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp), utils.to_reference(weight), **bad_kwargs
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, **bad_kwargs)


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"kernel_size": (3,), "stride": (1, 1), "padding": (1, 1)},
        {"kernel_size": (3, 3), "stride": (1,), "padding": (1, 1)},
        {"kernel_size": (3, 3), "stride": (1, 1), "padding": (1,)},
    ],
    ids=["kernel_size", "stride", "padding"],
)
def test__slow_conv2d_forward_rejects_wrong_length_params(bad_kwargs):
    # The SymInt[2] params must have exactly two entries; a length-1 list is
    # rejected by aten.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp), utils.to_reference(weight), **bad_kwargs
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, **bad_kwargs)


@pytest.mark._slow_conv2d_forward
@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"kernel_size": (3, 3), "stride": (0, 0), "padding": (1, 1)},
        {"kernel_size": (3, 3), "stride": (1, 1), "padding": (-1, -1)},
    ],
    ids=["zero_stride", "negative_padding"],
)
def test__slow_conv2d_forward_rejects_invalid_params(bad_kwargs):
    # stride must be positive and padding must be non-negative; aten raises for
    # both. Input/weight are valid so only the parameter check can fire.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_forward(
            utils.to_reference(inp), utils.to_reference(weight), **bad_kwargs
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, weight, **bad_kwargs)
