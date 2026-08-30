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

import os
import sys

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package (and, for the sibling benchmark file, ``benchmark``) resolve
# to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402
from .conftest import QUICK_MODE  # noqa: E402

# aten::slow_conv_transpose3d(Tensor self, Tensor weight, SymInt[3] kernel_size,
# Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0],
# SymInt[3] output_padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1]) -> Tensor is
# the im2col based "slow" transposed conv3d (groups always 1). ``self`` is
# (N, C_in, D, H, W) and ``weight`` is (C_in, C_out, kD, kH, kW) -- note the
# transposed layout: C_in first. The output is (N, C_out, D_out, H_out, W_out)
# with
#   D_out = (D - 1)*sD - 2*pD + dil_d*(kD - 1) + output_pad_d + 1
# and likewise for H and W. ``output_padding`` must be smaller than either
# ``stride`` or ``dilation`` along every dim. Each (input, weight, kernel_size,
# stride, padding, output_padding, dilation) tuple below is one distinct
# parametrized workload: they cover 1x1x1/2x2x2/3x3x3 kernels, stride 1/2 and
# asymmetric strides, padding 0/1/2 and asymmetric padding, output_padding 0/1,
# dilation 1/2, with and without bias. Element counts stay well below 1M so the
# correctness run stays fast. Inputs are generated through the value-range
# framework (tu.make_input) instead of torch.randn so the ranges are explicit
# and per-dtype.
if QUICK_MODE:
    SLOW_CONV_TRANSPOSE3D_CASES = [
        (
            (1, 2, 5, 5, 5),
            (2, 1, 3, 3, 3),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
        ),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    SLOW_CONV_TRANSPOSE3D_CASES = [
        (
            (1, 2, 5, 5, 5),
            (2, 1, 3, 3, 3),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (2, 3, 6, 6, 6),
            (3, 4, 3, 3, 3),
            (3, 3, 3),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (1, 3, 8, 8, 8),
            (3, 4, 3, 3, 3),
            (3, 3, 3),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        (
            (2, 4, 6, 6, 6),
            (4, 6, 3, 3, 3),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (2, 2, 2),
        ),
        (
            (1, 2, 7, 7, 7),
            (2, 3, 3, 3, 3),
            (3, 3, 3),
            (1, 1, 1),
            (2, 2, 2),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (2, 3, 5, 5, 5),
            (3, 5, 1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (2, 4, 5, 5, 5),
            (4, 3, 3, 3, 3),
            (3, 3, 3),
            (2, 1, 1),
            (1, 1, 0),
            (1, 0, 0),
            (1, 1, 1),
        ),
        (
            (1, 2, 4, 4, 4),
            (2, 3, 2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (2, 2, 2),
        ),
        (
            (2, 8, 4, 4, 4),
            (8, 4, 2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (1, 3, 9, 9, 9),
            (3, 2, 3, 3, 3),
            (3, 3, 3),
            (2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
        ),
        (
            (2, 2, 6, 5, 7),
            (2, 3, 3, 3, 3),
            (3, 3, 3),
            (1, 2, 1),
            (1, 1, 2),
            (0, 0, 0),
            (1, 1, 1),
        ),
        (
            (1, 2, 6, 6, 6),
            (2, 4, 3, 3, 3),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
        ),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)
    BIASES = [True, False]

# Value-range coverage: the transpose-conv output accumulates up to
# C_in*kD*kH*kW products, so the shared tu.selected_ranges() extremes
# (["0", "max"] / ["min", "0"]) overflow the fp16/bf16 accumulators. Use bounded
# local ranges that still span positive, negative and mixed-sign values; the
# shape/dtype/bias grid comes from the main parametrized cases above.
_VALUE_RANGE_CASES = (
    SLOW_CONV_TRANSPOSE3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_TRANSPOSE3D_CASES[0],  # 3x3x3 kernel, padding 1, dilation 1
        SLOW_CONV_TRANSPOSE3D_CASES[5],  # 1x1x1 kernel (pure GEMM path)
        SLOW_CONV_TRANSPOSE3D_CASES[3],  # dilation 2
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

# The aten op carries its own autograd (registered at the nn module level), so
# backward is exercised directly. Gradients are only validated on fp32/fp64:
# fp16/bf16 gradients accumulate too coarsely to compare against the analytic
# reference gradient.
_BACKWARD_CASES = (
    SLOW_CONV_TRANSPOSE3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_TRANSPOSE3D_CASES[0],  # padding 1, dilation 1
        SLOW_CONV_TRANSPOSE3D_CASES[1],  # stride 2
        SLOW_CONV_TRANSPOSE3D_CASES[3],  # dilation 2
    ]
)
_BACKWARD_DTYPES = [torch.float32] if QUICK_MODE else [torch.float32, torch.float64]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.slow_conv_transpose3d is registered; resolution order is:
    # (1) override, (2) the direct flag_gems.slow_conv_transpose3d callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_transpose3d", getattr(flag_gems, "slow_conv_transpose3d", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_transpose3d.out",
        getattr(flag_gems, "slow_conv_transpose3d_out", None),
    )


def _conv_transpose_output_shape(
    inp_shape, weight_shape, stride, padding, output_padding, dilation
):
    """(N, C_in, D, H, W) x (C_in, C_out, kD, kH, kW) -> (N, C_out, D_out, H_out, W_out)."""

    def _out_size(in_size, k, s, p, op, d):
        return (in_size - 1) * s - 2 * p + d * (k - 1) + op + 1

    return (inp_shape[0], weight_shape[1]) + tuple(
        _out_size(
            inp_shape[2 + i],
            weight_shape[2 + i],
            stride[i],
            padding[i],
            output_padding[i],
            dilation[i],
        )
        for i in range(3)
    )


def _make_conv_inputs(
    inp_shape, weight_shape, with_bias, dtype, value_range=("-1", "1")
):
    # Value-range framework instead of torch.randn: per-dtype explicit ranges.
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        # Transposed-conv bias has one element per output channel (the second
        # weight dim, C_out).
        bias = tu.make_input(dtype, (weight_shape[1],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # The reference is computed with an fp64 upcast, so it is exact for the
    # rounded inputs. The torch native op (and any good candidate) accumulates
    # the transpose-conv computation in the input dtype: fp16/bf16 tensor cores
    # keep at most fp16/bf16 precision per add, so the native op itself deviates
    # from the fp64 reference by up to ~7.8e-3 (fp16) and ~6.25e-2 (bf16) on the
    # larger 3-D reductions (up to C_in*kD*kH*kW = 96 terms). Measure the
    # deviation over multiple seeds and shapes: fp16 -> 2e-2 and bf16 -> 2e-1
    # give a ~2.5x margin; fp32 with TF32 disabled (set at the top of each test)
    # stays at ~2e-6, comfortably inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 2e-1
    elif dtype == torch.float16:
        atol = 2e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan, atol=atol)


@pytest.mark.slow_conv_transpose3d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE3D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose3d(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
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

    ref_out = torch.ops.aten.slow_conv_transpose3d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_transpose3d
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("value_range", _VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose3d_value_ranges(case, value_range, dtype, bias):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation = (
        case
    )
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    bias_t = tu.make_input(dtype, (weight_shape[1],), value_range) if bias else None
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_transpose3d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_transpose3d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE3D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose3d_out(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_transpose3d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_transpose3d.out(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
        out=ref_out,
    )
    assert ref_ret is ref_out

    out = torch.empty(ref_full.shape, dtype=dtype, device=flag_gems.device)
    res_ret = _resolve_gems_op_out()(
        inp,
        weight,
        kernel_size,
        bias_t,
        stride,
        padding,
        output_padding,
        dilation,
        out=out,
    )
    assert res_ret is out

    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_transpose3d_backward
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_slow_conv_transpose3d_backward(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation = (
        case
    )
    out_shape = _conv_transpose_output_shape(
        inp_shape, weight_shape, stride, padding, output_padding, dilation
    )

    inp = tu.make_input(dtype, inp_shape, ["-1", "1"])
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"])
    bias = tu.make_input(dtype, (weight_shape[1],), ["-1", "1"])
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    # Reference graph on the fp64-upcast inputs.
    ref_inp = utils.to_reference(inp, True).requires_grad_()
    ref_weight = utils.to_reference(weight, True).requires_grad_()
    ref_bias = utils.to_reference(bias, True).requires_grad_()
    ref_grad_out = utils.to_reference(grad_out, True)

    ref_out = torch.ops.aten.slow_conv_transpose3d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    )
    ref_gi, ref_gw, ref_gb = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )

    # Self-check: the low-level op's autograd must match the standard
    # F.conv_transpose3d backward (same math, im2col vs direct formulation).
    f_out = torch.nn.functional.conv_transpose3d(
        ref_inp,
        ref_weight,
        ref_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    f_gi, f_gw, f_gb = torch.autograd.grad(
        f_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )
    tu.assert_result_close(ref_gi, f_gi)
    tu.assert_result_close(ref_gw, f_gw)
    tu.assert_result_close(ref_gb, f_gb)

    # The candidate forward must match the fp64 reference...
    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, stride, padding, output_padding, dilation
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


@pytest.mark.slow_conv_transpose3d_nan_inf
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_transpose3d_nan_inf(dtype):
    # A single nan and a single inf in the input, with a positive unit 1x1x1
    # kernel and a unit bias: each output element is the sum of exactly one
    # input element and one bias term, so the nan/inf land at exactly the same
    # output positions in the reference and any faithful candidate (no
    # inf/-inf cancellation).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = torch.ones((1, 1, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    inp[0, 0, 1, 1, 1] = float("nan")
    inp[0, 0, 2, 2, 2] = float("inf")
    weight = torch.ones((1, 1, 1, 1, 1), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((1,), dtype=dtype, device=flag_gems.device)
    kernel_size = (1, 1, 1)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_transpose3d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        (1, 1, 1),
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1)
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_output_padding_not_less_than_stride():
    # output_padding must be smaller than either stride or dilation along every
    # dim; here output_padding == stride along all three dims.
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (2, 1, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)
        )


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_negative_stride():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (2, 1, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_negative_dilation():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (2, 1, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (-1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (-1, 1, 1)
        )


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_int_dtype():
    # Only floating dtypes are implemented for the transposed conv3d.
    inp = tu.make_input(torch.int32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.int32, (2, 1, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_6d_input():
    # self must be (N, C_in, D, H, W) (5D batch mode) or (C_in, D, H, W) (4D);
    # a 6D input is rejected.
    inp = tu.make_input(torch.float32, (1, 2, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 1, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )


@pytest.mark.slow_conv_transpose3d_negative
def test_slow_conv_transpose3d_rejects_4d_weight():
    # weight must be 5D (C_in, C_out, kD, kH, kW); a 4D weight is rejected.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 1, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _resolve_gems_op()(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)
        )
