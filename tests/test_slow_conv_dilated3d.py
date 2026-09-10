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

# aten::slow_conv_dilated3d(Tensor self, Tensor weight, SymInt[3] kernel_size,
# Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0],
# SymInt[3] dilation=[1, 1, 1]) -> Tensor is the im2col based "slow" conv3d with
# dilation support (groups always 1). ``self`` is (N, C_in, D, H, W), ``weight``
# is (C_out, C_in, kD, kH, kW) and ``kernel_size`` must match the weight spatial
# dims. The output is (N, C_out, D_out, H_out, W_out) with
#   D_out = (D + 2*pD - dil_d*(kD - 1) - 1) // sD + 1
# and likewise for H and W.
#
# Dtype probe (tu.supported_dtypes / direct ATen calls on the active device):
# the CUDA kernel is implemented only for the floating dtypes fp16 / fp32 /
# bf16 / fp64 -- int8, uint8, float8_e4m3fn, float8_e5m2, int32 and int64 all
# raise `"slow_conv_dilated<>" not implemented for '<Int/Float8...>'`. The
# correctness grid therefore uses utils.ALL_FLOAT_DTYPES; integer/fp8 rejection
# is covered by the negative dtype test below instead of being silently
# dropped.
#
# Rank probe: although the schema also accepts an unbatched 4-D input
# (C_in, D, H, W), the CUDA reference kernel is non-deterministic on that path
# (two identical ATen calls on identical fp64 inputs disagree, and the result
# does not match F.conv3d), so it cannot serve as an oracle. Only the reliable
# 5-D batched route is exercised; the 6-D rejection test covers the rank check.
#
# Shape coverage: the operator is rank-fixed and structurally constrained
# (C_in must agree between input and weight, kernel_size must agree with the
# weight), so the generic tu.selected_shapes() set cannot be applied directly.
# The explicit (input, weight, kernel_size, stride, padding, dilation) tuples
# below play the role of the shape levels: they span 1x1x1/2x2x2/3x3x3 kernels,
# stride 1/2, padding 0/1/2, dilation 1/2, asymmetric strides/paddings, and
# with/without bias. Element counts stay well below 1M so the correctness run
# stays fast. Inputs are generated through the value-range framework
# (tu.make_input) instead of torch.randn so the ranges are explicit and
# per-dtype.
if QUICK_MODE:
    SLOW_CONV_DILATED3D_CASES = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    SLOW_CONV_DILATED3D_CASES = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((2, 3, 6, 6, 6), (4, 3, 3, 3, 3), (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 3, 8, 8, 8), (4, 3, 3, 3, 3), (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
        ((2, 4, 6, 6, 6), (6, 4, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2)),
        ((1, 2, 7, 7, 7), (3, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ((2, 3, 5, 5, 5), (5, 3, 1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((2, 4, 5, 5, 5), (3, 4, 3, 3, 3), (3, 3, 3), (2, 1, 1), (1, 1, 0), (1, 1, 1)),
        ((1, 2, 4, 4, 4), (3, 2, 2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 2, 2)),
        ((2, 8, 4, 4, 4), (4, 8, 2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 3, 9, 9, 9), (2, 3, 3, 3, 3), (3, 3, 3), (2, 2, 2), (2, 2, 2), (1, 1, 1)),
        ((2, 2, 6, 5, 7), (3, 2, 3, 3, 3), (3, 3, 3), (1, 2, 1), (1, 1, 2), (1, 1, 1)),
        ((1, 2, 6, 6, 6), (4, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)
    BIASES = [True, False]

# Value-range coverage: the conv reduction sums up to C_in*kD*kH*kW = 108
# products of independently drawn input/weight values, so the shared
# tu.selected_ranges() extremes (["0", "max"] / ["min", "0"]) overflow every
# floating accumulator (e.g. fp32 max^2 = inf). Use bounded local ranges that
# still span mixed-sign, non-negative and non-positive values; the
# shape/dtype/bias grid comes from the main parametrized cases above. This is
# the documented per-operator adaptation the spec allows for multiplicative
# reductions.
_VALUE_RANGE_CASES = (
    SLOW_CONV_DILATED3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_DILATED3D_CASES[0],  # 3x3x3 kernel, padding 1, dilation 1
        SLOW_CONV_DILATED3D_CASES[5],  # 1x1x1 kernel (pure GEMM path)
        SLOW_CONV_DILATED3D_CASES[3],  # dilation 2
    ]
)
_VALUE_RANGES = (
    [["-1", "1"]]
    if QUICK_MODE
    else [
        ["-1", "1"],  # mixed signs (cancellation)
        ["0", "1"],  # non-negative
        ["-1", "0"],  # non-positive
    ]
)

# The aten op carries its own autograd (SlowConvDilated3DBackward0), so
# backward is exercised directly. Gradients are only validated on fp32/fp64:
# fp16/bf16 gradients accumulate too coarsely to compare against the analytic
# reference gradient.
_BACKWARD_CASES = (
    SLOW_CONV_DILATED3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_DILATED3D_CASES[0],  # padding 1, dilation 1
        SLOW_CONV_DILATED3D_CASES[1],  # no padding
        SLOW_CONV_DILATED3D_CASES[3],  # dilation 2
    ]
)
_BACKWARD_DTYPES = [torch.float32] if QUICK_MODE else [torch.float32, torch.float64]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.slow_conv_dilated3d is registered; resolution order is:
    # (1) override, (2) the direct flag_gems.slow_conv_dilated3d callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_dilated3d", getattr(flag_gems, "slow_conv_dilated3d", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_dilated3d.out", getattr(flag_gems, "slow_conv_dilated3d_out", None)
    )


def _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation):
    """(N, C_in, D, H, W) x (C_out, C_in, kD, kH, kW) -> (N, C_out, D_out, H_out, W_out)."""

    def _out_size(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    spatial = tuple(
        _out_size(
            inp_shape[2 + i], weight_shape[2 + i], stride[i], padding[i], dilation[i]
        )
        for i in range(3)
    )
    return (inp_shape[0], weight_shape[0]) + spatial


def _make_conv_inputs(
    inp_shape, weight_shape, with_bias, dtype, value_range=("-1", "1")
):
    # Value-range framework instead of torch.randn: per-dtype explicit ranges.
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        # bias has one element per output channel (the first weight dim).
        bias = tu.make_input(dtype, (weight_shape[0],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # The reference is computed with an fp64 upcast, so it is exact for the
    # rounded inputs. The torch native op (and any good candidate) accumulates
    # the im2col GEMM in the input dtype: fp16/bf16 tensor cores keep at most
    # fp16/bf16 precision per add, so the native op itself deviates from the
    # fp64 reference by up to ~8e-3 (fp16) and ~1.3e-1 (bf16) on the larger
    # 3D reductions (up to C_in*kD*kH*kW = 108 terms). Measured over multiple
    # seeds and shapes: fp16 -> 2e-2 and bf16 -> 2e-1 give a comfortable margin
    # (flag_gems.testing.assert_close also adds rtol=1e-3 / 0.016), while fp32
    # with TF32 disabled (set at the top of each test) stays at ~1e-5,
    # comfortably inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 2e-1
    elif dtype == torch.float16:
        atol = 2e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan, atol=atol)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED3D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d(
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

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(inp, weight, kernel_size, bias_t, stride, padding, dilation)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("value_range", _VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d_value_ranges(case, value_range, dtype, bias):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    bias_t = tu.make_input(dtype, (weight_shape[0],), value_range) if bias else None
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated3d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED3D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_dilated3d.out(
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

    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_dilated3d_backward
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_slow_conv_dilated3d_backward(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    out_shape = _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, inp_shape, ["-1", "1"])
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"])
    bias = tu.make_input(dtype, (weight_shape[0],), ["-1", "1"])
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    # Reference graph on the fp64-upcast inputs.
    ref_inp = utils.to_reference(inp, True).requires_grad_()
    ref_weight = utils.to_reference(weight, True).requires_grad_()
    ref_bias = utils.to_reference(bias, True).requires_grad_()
    ref_grad_out = utils.to_reference(grad_out, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_gi, ref_gw, ref_gb = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )

    # Self-check: the low-level op's autograd must match the standard
    # F.conv3d backward (same math, im2col vs direct formulation).
    f_out = torch.nn.functional.conv3d(
        ref_inp, ref_weight, ref_bias, stride=stride, padding=padding, dilation=dilation
    )
    f_gi, f_gw, f_gb = torch.autograd.grad(
        f_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )
    tu.assert_result_close(ref_gi, f_gi)
    tu.assert_result_close(ref_gw, f_gw)
    tu.assert_result_close(ref_gb, f_gb)

    # The candidate forward must match the fp64 reference...
    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )
    _assert_close(res_out, ref_out.to(dtype), dtype)

    # ...and, if the candidate kernel is autograd-aware, its gradients must
    # match the reference gradients too.
    if res_out.requires_grad:
        res_gi, res_gw, res_gb = torch.autograd.grad(
            res_out, (inp, weight, bias), grad_outputs=grad_out
        )
        _assert_close(res_gi, ref_gi.to(dtype), dtype)
        _assert_close(res_gw, ref_gw.to(dtype), dtype)
        _assert_close(res_gb, ref_gb.to(dtype), dtype)


@pytest.mark.slow_conv_dilated3d_nan_inf
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_dilated3d_nan_inf(dtype):
    # A single nan and a single inf in the input, with a positive unit-weight
    # kernel: every window sum is either a small exact integer, nan (window
    # touches the nan) or inf (window touches the inf), with no inf/-inf
    # cancellation, so the reference and any faithful candidate must place the
    # nan/inf at exactly the same output positions.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = torch.ones((1, 1, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    inp[0, 0, 1, 1, 1] = float("nan")
    inp[0, 0, 2, 2, 2] = float("inf")
    weight = torch.ones((1, 1, 2, 2, 2), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((1,), dtype=dtype, device=flag_gems.device)
    kernel_size = (2, 2, 2)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_wrong_kernel_size():
    # kernel_size must match the weight spatial dims exactly.
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (2, 2, 2), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (2, 2, 2), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_channel_mismatch():
    # input.size(1) (C_in) must equal weight.size(1).
    inp = tu.make_input(torch.float32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 3, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_int_dtype():
    # Only floating dtypes are implemented for the conv3d; int8/uint8/fp8/int32
    # all raise on the native path.
    for bad_dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.int32):
        inp = tu.make_input(bad_dtype, (1, 2, 5, 5, 5), ["-1", "1"])
        weight = tu.make_input(bad_dtype, (1, 2, 3, 3, 3), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten.slow_conv_dilated3d(
                inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            _resolve_gems_op()(
                inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_6d_input():
    # self must be a 5-D (N, C_in, D, H, W) tensor; a 6-D input is rejected.
    inp = tu.make_input(torch.float32, (1, 2, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_wrong_weight_rank():
    # weight must be 5-D (C_out, C_in, kD, kH, kW); a 4-D weight is rejected.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_negative_stride():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_negative_dilation():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (-1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (-1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_output_size_too_small():
    # Dilation so large that the computed output size becomes negative.
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (5, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (5, 1, 1)
        )
