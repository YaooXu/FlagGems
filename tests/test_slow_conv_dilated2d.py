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

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu
from .conftest import QUICK_MODE

# aten::slow_conv_dilated2d(Tensor self, Tensor weight, SymInt[2] kernel_size,
# Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0],
# SymInt[2] dilation=[1, 1]) -> Tensor is the im2col based "slow" conv2d with
# dilation support (groups always 1). ``self`` is (N, C_in, H, W), ``weight`` is
# (C_out, C_in, kH, kW) and ``kernel_size`` must match the weight spatial dims.
# The output is (N, C_out, H_out, W_out) with
#   H_out = (H + 2*pH - dil_h*(kH - 1) - 1) // sH + 1
# and likewise for W. Each (input, weight, kernel_size, stride, padding,
# dilation) tuple below is one distinct parametrized workload: they cover
# 1x1/2x2/2x3/3x3/3x5/5x5 kernels, stride 1/2/3 and asymmetric strides, padding
# 0/1/2 and asymmetric padding, dilation 1/2 and asymmetric dilation, with and
# without bias. The canonical tu.selected_shapes() set is pointwise-shaped and
# cannot describe a conv (whose input must be 4-D), so these local tuples play
# the role of the shape levels. Element counts stay well below 1M so the
# correctness run stays fast.
#
# Dtype coverage (常规算子测试用例): a direct ATen call on the active device
# (the same probe the negative dtype test below relies on) shows the CUDA
# kernel only implements the floating types -- int8 / uint8 /
# float8_e4m3fn / float8_e5m2 / int32 / int64 all raise
# RuntimeError("slow_conv_dilated<>" not implemented for ...). The required
# int8/uint8/fp8 grid therefore does not apply to this operator; the file covers
# every supported float dtype via utils.ALL_FLOAT_DTYPES (fp16 / fp32 / bf16,
# plus fp64 where the backend supports it), and the unsupported dtypes are
# asserted to raise in test_slow_conv_dilated2d_rejects_unsupported_dtype.
# Broadcasting is not a meaningful dimension for a convolution (input and weight
# shapes are independent), so it is intentionally omitted.
if QUICK_MODE:
    SLOW_CONV_DILATED2D_CASES = [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    SLOW_CONV_DILATED2D_CASES = [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 9, 9), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 3, 8, 8), (5, 3, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 4, 8, 8), (6, 4, 3, 3), (3, 3), (1, 1), (2, 2), (2, 2)),
        ((1, 3, 7, 9), (4, 3, 3, 5), (3, 5), (1, 1), (1, 2), (1, 1)),
        ((2, 8, 16, 16), (16, 8, 1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
        ((1, 4, 12, 12), (8, 4, 5, 5), (5, 5), (2, 2), (2, 2), (2, 2)),
        ((2, 3, 6, 10), (5, 3, 3, 3), (3, 3), (2, 1), (1, 2), (1, 2)),
        ((2, 16, 12, 12), (8, 16, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 2, 4, 4), (3, 2, 2, 2), (2, 2), (1, 1), (0, 0), (2, 2)),
        # asymmetric kernel (2x3) and stride 3
        ((2, 3, 10, 10), (4, 3, 2, 3), (2, 3), (1, 1), (0, 0), (1, 1)),
        # stride 3 with asymmetric dilation (2, 1)
        ((1, 2, 9, 9), (3, 2, 3, 3), (3, 3), (3, 3), (0, 0), (2, 1)),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)
    BIASES = [True, False]

# Regular-operator value-range coverage (常规算子测试用例): inputs are drawn from
# tu.make_input over the shared ranges, scaled by _INPUT_SCALE. The reference is
# an fp64 upcast of the exact same values, so the comparison isolates real
# indexing/formula bugs. A modest scale keeps the fp16/bf16 im2col-GEMM
# reduction noise (which grows with data magnitude) well inside the calibrated
# _assert_close tolerances. Constant fills are added as extra exact-value
# workloads on top of the five spec ranges returned by tu.selected_ranges().
_INPUT_SCALE = 0.1
_SLOW_CONV_DILATED2D_VALUE_RANGES = tu.selected_ranges() + [
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
]
if QUICK_MODE:
    _SLOW_CONV_DILATED2D_VALUE_RANGES_CASES = [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ]
else:
    _SLOW_CONV_DILATED2D_VALUE_RANGES_CASES = [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 4, 6, 6), (4, 4, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2)),
        ((2, 2, 6, 6), (3, 2, 1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
    ]

# Backward coverage: the aten op carries autograd, so the fp64 reference
# gradients come from torch.autograd.grad over the native op. fp16/bf16
# gradients accumulate too coarsely to compare against the analytic reference,
# so only fp32/fp64 are validated.
if QUICK_MODE:
    _SLOW_CONV_DILATED2D_BACKWARD_CASES = [
        ((1, 2, 5, 5), (3, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ]
    _BACKWARD_DTYPES = [torch.float32]
else:
    _SLOW_CONV_DILATED2D_BACKWARD_CASES = [
        ((1, 2, 5, 5), (3, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 6, 6), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    ]
    _BACKWARD_DTYPES = [torch.float32, torch.float64]

# Invalid configurations for the negative tests, as
# (inp_shape, weight_shape, kernel_size, stride, padding, dilation): channel
# mismatches (C_in/C_out), kernel_size disagreeing with the weight spatial dims,
# non-4-D inputs, zero stride and dilation so large the kernel cannot fit.
# Negative padding is deliberately NOT here: the native op accepts it (it is a
# valid, degenerate configuration).
_INVALID_SLOW_CONV_DILATED2D_CASES = [
    # weight C_in disagrees with input C_in
    ((2, 3, 5, 5), (4, 5, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # weight spatial dims disagree with kernel_size
    ((2, 3, 5, 5), (4, 3, 4, 4), (3, 3), (1, 1), (0, 0), (1, 1)),
    # input is not 4-D
    ((2, 3, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # weight is not 4-D
    ((2, 3, 5, 5), (4, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # kernel_size spatial dims disagree with the weight spatial dims
    ((2, 3, 5, 5), (4, 3, 3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
    # zero stride
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (0, 0), (0, 0), (1, 1)),
    # zero dilation
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (0, 0)),
    # dilation so large the kernel no longer fits inside the input
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (10, 10)),
]

# The CUDA kernel only implements floating dtypes (probe result); these must be
# rejected. Tensors are built with torch.ones because make_input cannot produce
# negative values for the unsigned dtypes.
_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.slow_conv_dilated2d is registered; resolution order is:
    # (1) override, (2) the direct flag_gems.slow_conv_dilated2d callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_dilated2d", getattr(flag_gems, "slow_conv_dilated2d", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_dilated2d.out", getattr(flag_gems, "slow_conv_dilated2d_out", None)
    )


def _make_conv_inputs(
    inp_shape, weight_shape, with_bias, dtype, value_range=("-1", "1")
):
    """Build (input, weight, bias) from the value-range framework (tu.make_input),
    scaled so the fp16/bf16 im2col-GEMM reduction noise stays inside the
    _assert_close tolerance (the fp64 reference is exact for the rounded
    values)."""
    inp = _INPUT_SCALE * tu.make_input(dtype, inp_shape, value_range)
    weight = _INPUT_SCALE * tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = _INPUT_SCALE * tu.make_input(dtype, (weight_shape[0],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # The reference is computed with an fp64 upcast, so it is exact for the
    # rounded inputs. The torch native op (and any good candidate) accumulates
    # the im2col GEMM in the input dtype: fp16/bf16 tensor cores keep at most
    # fp16/bf16 precision per add, so the native op itself deviates from the
    # fp64 reference by up to ~2e-3 (fp16) on the larger reductions. Measure
    # the deviation over multiple seeds and shapes: fp16 -> 1e-2 and bf16 ->
    # 5e-2 give a wide margin; fp32 with TF32 disabled (set at the top of each
    # test) stays at ~2e-5, comfortably inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 5e-2
    elif dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan, atol=atol)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED2D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    # The reference op runs cuBLAS/baddbmm for the im2col GEMM; keep TF32 off so
    # the fp32 comparison stays at the standard 1e-4 tolerance (with TF32 on the
    # native op itself deviates from the fp64 reference by ~1.6e-2).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(inp, weight, kernel_size, bias_t, stride, padding, dilation)

    assert res_out.shape == ref_out.shape
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED2D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_dilated2d.out(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        dilation,
        out=ref_out,
    )
    assert ref_ret is ref_out

    out = torch.empty(ref_full.shape, dtype=dtype, device=flag_gems.device)
    res_ret = _resolve_gems_op_out()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation, out=out
    )
    assert res_ret is out
    assert res_ret.dtype == dtype
    assert res_ret.shape == ref_ret.shape

    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _SLOW_CONV_DILATED2D_VALUE_RANGES_CASES,
)
@pytest.mark.parametrize("value_range", _SLOW_CONV_DILATED2D_VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d_value_ranges(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    dilation,
    value_range,
    dtype,
    bias,
):
    # Value-range coverage: the same configuration must produce the same output
    # for every value range the op can see (positive-only, negative-only, mixed,
    # and constant 0/1/-1 fills). This replaces the plain-randn value tests with
    # a deterministic, per-dtype value-range framework.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, value_range
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated2d_backward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _SLOW_CONV_DILATED2D_BACKWARD_CASES,
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_slow_conv_dilated2d_backward(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype
):
    # Backward coverage (常规算子测试用例): the fp64 reference gradients come from
    # torch.autograd.grad over the native op; the candidate forward must match
    # the reference output, and, when the candidate kernel advertises autograd
    # support, its gradients must match the reference gradients too (with atol
    # scaled by the contraction size of each gradient).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = _INPUT_SCALE * tu.make_input(dtype, inp_shape, ["-1", "1"]).requires_grad_()
    weight = (
        _INPUT_SCALE * tu.make_input(dtype, weight_shape, ["-1", "1"]).requires_grad_()
    )
    bias = (
        _INPUT_SCALE
        * tu.make_input(dtype, (weight_shape[0],), ["-1", "1"]).requires_grad_()
    )

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_fwd = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
        ref_fwd.sum(), (ref_inp, ref_weight, ref_bias)
    )

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )
    _assert_close(res_out, ref_fwd.to(dtype), dtype)

    # grad_input contracts over C_out x kH x kW; grad_weight/grad_bias contract
    # over N x H_out x W_out. Scale atol by those counts.
    in_reduce_dim = weight_shape[0] * weight_shape[2] * weight_shape[3]
    out_reduce_dim = inp_shape[0] * ref_fwd.shape[2] * ref_fwd.shape[3]

    if res_out.requires_grad:
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out.sum(), (inp, weight, bias)
        )
        utils.gems_assert_close(
            res_in_grad, ref_in_grad, dtype, reduce_dim=in_reduce_dim
        )
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=out_reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=out_reduce_dim
        )


@pytest.mark.slow_conv_dilated2d_nan_inf
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_dilated2d_nan_inf(dtype):
    # nan/inf/-inf propagate deterministically through a unit 1x1 kernel: each
    # output element is exactly inp[0, 0, i, j] + inp[0, 1, i, j] + 1, so the
    # nan/inf/-inf land at exactly the same output positions on both paths and
    # no inf + (-inf) cancellation can occur. equal_nan=True tolerates the nan
    # entries while inf must still match (an inf-vs-nan mismatch fails the
    # compare).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = torch.ones((1, 2, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((1, 2, 1, 1), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((1,), dtype=dtype, device=flag_gems.device)
    inp[0, 0, 1, 1] = float("nan")
    inp[0, 1, 2, 2] = float("inf")
    inp[0, 0, 3, 3] = float("-inf")

    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.slow_conv_dilated2d_negative
@pytest.mark.parametrize("case", _INVALID_SLOW_CONV_DILATED2D_CASES)
def test_slow_conv_dilated2d_negative(case):
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    inp, weight, bias = _make_conv_inputs(inp_shape, weight_shape, True, torch.float32)

    # The reference rejects the inconsistent configuration...
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated2d(
            inp, weight, kernel_size, bias, stride, padding, dilation
        )

    # ...and so must the candidate. LookupError is tolerated only when no
    # candidate has been injected for this run.
    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(inp, weight, kernel_size, bias, stride, padding, dilation)


@pytest.mark.slow_conv_dilated2d_negative
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_slow_conv_dilated2d_rejects_unsupported_dtype(dtype):
    # The CUDA kernel only implements floating dtypes; int8 / uint8 / fp8 /
    # int32 / int64 must be rejected by both the reference and the candidate.
    inp = torch.ones((2, 3, 5, 5), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((4,), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated2d(
            inp, weight, (3, 3), bias, (1, 1), (0, 0), (1, 1)
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(inp, weight, (3, 3), bias, (1, 1), (0, 0), (1, 1))
