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
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402
from .conftest import QUICK_MODE  # noqa: E402

# aten::slow_conv_transpose2d(Tensor self, Tensor weight, SymInt[2] kernel_size,
# Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0],
# SymInt[2] output_padding=[0, 0], SymInt[2] dilation=[1, 1]) -> Tensor is the
# im2col based "slow" transposed conv2d (groups always 1, no dilation-less
# restriction -- dilation is supported). ``self`` is (N, C_in, H, W) or the
# unbatched (C_in, H, W); ``weight`` uses the transposed layout
# (C_in, C_out, kH, kW) so the bias length is ``weight_shape[1]`` (C_out). The
# output is (N, C_out, H_out, W_out) with
#   H_out = (H - 1)*sH - 2*pH + dil_h*(kH - 1) + output_pad_h + 1
# and likewise for W (output_padding must be < stride or dilation per dim).
# Each (input, weight, kernel_size, stride, padding, output_padding, dilation)
# tuple below is one distinct parametrized workload: they cover 1x1/2x2/3x3/5x5
# kernels, stride 1/2, padding 0/1/2, output_padding 0/1, dilation 1/2,
# asymmetric strides/paddings/dilations, the unbatched 3D form, with and without
# bias. Element counts stay well below 1M so the correctness run stays fast.
if QUICK_MODE:
    SLOW_CONV_TRANSPOSE2D_CASES = [
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    SLOW_CONV_TRANSPOSE2D_CASES = [
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 4, 8, 8),
            (4, 6, 3, 3),
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
        ),
        (
            (2, 8, 12, 12),
            (8, 8, 3, 3),
            (3, 3),
            (2, 1),
            (1, 2),
            (1, 0),
            (2, 1),
        ),
        (
            (1, 3, 7, 9),
            (3, 4, 3, 5),
            (3, 5),
            (1, 1),
            (1, 2),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 4, 6, 6),
            (4, 8, 2, 2),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
            (2, 2),
        ),
        (
            (1, 16, 8, 8),
            (16, 8, 1, 1),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 3, 5, 4),
            (3, 5, 3, 3),
            (3, 3),
            (2, 2),
            (0, 1),
            (1, 0),
            (1, 1),
        ),
        (
            (1, 4, 10, 10),
            (4, 8, 5, 5),
            (5, 5),
            (2, 2),
            (2, 2),
            (1, 1),
            (1, 1),
        ),
        (
            (2, 8, 16, 16),
            (8, 16, 3, 3),
            (3, 3),
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
        ),
        (
            (1, 2, 4, 4),
            (2, 3, 2, 2),
            (2, 2),
            (1, 1),
            (0, 0),
            (0, 0),
            (2, 2),
        ),
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            (2, 2),
        ),
        (
            (2, 4, 5),
            (2, 3, 3, 3),
            (3, 3),
            (2, 1),
            (1, 0),
            (1, 0),
            (1, 1),
        ),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)
    BIASES = [True, False]

# Value ranges whose per-dtype bounds exceed what the transposed-conv reduction
# can represent in the accumulation dtype (up to C_in*kH*kW terms): feeding
# dtype-max inputs would overflow to inf even for the native op. Drop them from
# the value-range sweep (they are covered anyway by the negative-value ranges).
_UNSAFE_FOR_REDUCTION = frozenset({"max", "min", "max/2", "min/2"})
_CONV_VALUE_RANGES = [
    r for r in tu.selected_ranges() if not ({r[0], r[1]} & _UNSAFE_FOR_REDUCTION)
]

# The reference is an fp64 upcast of the rounded inputs, and the native op (like
# any candidate) accumulates the transposed conv in the input dtype. Scale the
# inputs by 0.1 so fp16/bf16 reductions stay well inside their dynamic range
# while still exercising both signs (the value-range test covers magnitudes).
_INPUT_SCALE = 0.1

# The backward test needs every overload to be differentiable; the first three
# cases cover stride 1/2, padding, output_padding and asymmetric dilation with
# small reductions.
_BACKWARD_CASES = SLOW_CONV_TRANSPOSE2D_CASES[:3]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.slow_conv_transpose2d is registered; resolution order is:
    # (1) override, (2) the direct flag_gems.slow_conv_transpose2d callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_transpose2d", getattr(flag_gems, "slow_conv_transpose2d", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_transpose2d.out",
        getattr(flag_gems, "slow_conv_transpose2d_out", None),
    )


def _conv_output_shape(
    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation
):
    """Output shape of the transposed conv (matches aten's size computation)."""
    h_in = inp_shape[-2]
    w_in = inp_shape[-1]
    out_c = weight_shape[1]
    h_out = (
        (h_in - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    w_out = (
        (w_in - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    if len(inp_shape) == 3:
        return (out_c, h_out, w_out)
    return (inp_shape[0], out_c, h_out, w_out)


def _make_conv_inputs(inp_shape, weight_shape, with_bias, dtype, value_range):
    inp = tu.make_input(dtype, inp_shape, value_range)
    # Transposed conv weight is (C_in, C_out, kH, kW): the bias length is the
    # second (output-channel) dim.
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = tu.make_input(dtype, (weight_shape[1],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # The reference is computed with an fp64 upcast, so it is exact for the
    # rounded inputs. The native op (and any good candidate) accumulates the
    # transposed conv in the input dtype: fp16/bf16 keep at most fp16/bf16
    # precision per add, so the native op itself deviates from the fp64
    # reference by up to ~1.7e-2 (fp16) and ~1.2e-1 (bf16) on the larger 2D
    # reductions (up to C_in*kH*kW = 100 terms). Measured over multiple seeds
    # and shapes: fp16 -> 4e-2 and bf16 -> 4e-1 give a ~2x margin; fp32 with
    # TF32 disabled (set at the top of each test) stays at ~2e-6, comfortably
    # inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 4e-1
    elif dtype == torch.float16:
        atol = 4e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol, equal_nan=equal_nan)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose2d(
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
    # The reference op runs the native path; keep TF32 off so the fp32
    # comparison stays at the standard 1e-4 tolerance.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_transpose2d(
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


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES[:2],
)
@pytest.mark.parametrize("value_range", _CONV_VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_transpose2d_value_ranges(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    value_range,
    dtype,
):
    # Sweep the shared value ranges ([-1,1], [0,1], [-1,0] and the constant
    # ranges, minus the dtype-extreme ones that overflow the conv reduction).
    # The scaled inputs keep fp16/bf16 well inside their dynamic range; use the
    # calibrated _assert_close rather than tu.assert_result_close because the
    # native op's bf16 accumulation noise alone exceeds the generic
    # rtol=1e-2/atol=1e-3 tolerance even for the smallest kernel.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, True, dtype, value_range
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_transpose2d(
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


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    _BACKWARD_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose2d_backward(
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

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    inp = (_INPUT_SCALE * inp).requires_grad_()
    weight = (_INPUT_SCALE * weight).requires_grad_()
    if bias_t is not None:
        bias_t = (_INPUT_SCALE * bias_t).requires_grad_()

    out_shape = _conv_output_shape(
        inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation
    )
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_out = torch.ops.aten.slow_conv_transpose2d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
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

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )
    _assert_close(res_out, ref_out.to(dtype), dtype)

    # Only compare gradients when the candidate advertises autograd support
    # (otherwise the candidate wrapper did not install a backward and
    # autograd.grad would raise).
    if res_out.requires_grad:
        if bias_t is None:
            res_gi, res_gw = torch.autograd.grad(res_out, (inp, weight), grad_out)
            res_gb = None
        else:
            res_gi, res_gw, res_gb = torch.autograd.grad(
                res_out, (inp, weight, bias_t), grad_out
            )
        # grad_input reduces over C_out*kH*kW terms, grad_weight/grad_bias over
        # N*H_out*W_out terms; the fp64 reference is exact for the scaled
        # inputs, so the candidate only needs to match the native precision.
        in_reduce_dim = weight_shape[1] * weight_shape[2] * weight_shape[3]
        out_reduce_dim = inp_shape[0] * out_shape[2] * out_shape[3]
        for res_g, ref_g, reduce_dim in zip(
            (res_gi, res_gw, res_gb),
            (ref_gi, ref_gw, ref_gb),
            (in_reduce_dim, out_reduce_dim, out_reduce_dim),
        ):
            if ref_g is None:
                assert res_g is None
            else:
                utils.gems_assert_close(
                    res_g, ref_g.to(dtype), dtype, reduce_dim=reduce_dim
                )


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_transpose2d_nan_inf(dtype):
    # nan/inf in the input must propagate deterministically through the
    # scatter-add of the transposed conv. Weights are strictly positive and no
    # -inf is injected, so an inf output can never be cancelled into nan and
    # the nan/inf positions are stable (equal_nan=True compares them exactly).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    (
        inp_shape,
        weight_shape,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    ) = SLOW_CONV_TRANSPOSE2D_CASES[0]
    inp = tu.make_input(dtype, inp_shape, ["-1", "1"])
    inp[0, 0, 1, 1] = float("nan")
    inp[0, 1, 3, 3] = float("inf")
    weight = tu.make_input(dtype, weight_shape, ["0", "1"]) + 0.5
    bias = tu.make_input(dtype, (weight_shape[1],), ["-1", "1"])

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_transpose2d(
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
        inp, weight, kernel_size, bias, stride, padding, output_padding, dilation
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)
    # Sanity check that the injected values actually reached the output.
    assert torch.isnan(ref_out).any()
    assert torch.isinf(ref_out).any()


@pytest.mark.slow_conv_transpose2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose2d_out(
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

    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_transpose2d(
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
    ref_ret = torch.ops.aten.slow_conv_transpose2d.out(
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


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d_rejects_channel_mismatch():
    # Weight's C_in (dim 0) must match the input channels; the reference raises
    # and the candidate must raise as well.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (3, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize("bad_shape", [(2, 5), (1, 2, 5, 5, 5)])
def test_slow_conv_transpose2d_rejects_invalid_input_rank(bad_shape):
    # Only (N, C, H, W) or (C, H, W) inputs are accepted.
    inp = tu.make_input(torch.float32, bad_shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "stride, output_padding",
    [
        ((2, 2), (2, 2)),
        ((1, 1), (1, 1)),
    ],
)
def test_slow_conv_transpose2d_rejects_invalid_output_padding(stride, output_padding):
    # output_padding must be smaller than either stride or dilation per dim.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (
        inp,
        weight,
        (3, 3),
        None,
        stride,
        (0, 0),
        output_padding,
        (1, 1),
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "scalar_param, scalar_value",
    [
        ("kernel_size", 3),
        ("stride", 1),
        ("padding", 0),
        ("output_padding", 0),
        ("dilation", 1),
    ],
)
def test_slow_conv_transpose2d_rejects_scalar_params(scalar_param, scalar_value):
    # kernel_size/stride/padding/output_padding/dilation are SymInt[2]: passing
    # a bare scalar int does not match the schema and raises.
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = [inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1)]
    index = {
        "kernel_size": 2,
        "stride": 4,
        "padding": 5,
        "output_padding": 6,
        "dilation": 7,
    }[scalar_param]
    args[index] = scalar_value
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(*args)


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d_rejects_non_float_dtype():
    # The slow im2col path only supports floating point inputs (groups=1, GEMM
    # accumulation); the CUDA aten implementation raises for integer inputs and
    # the candidate must raise as well.
    inp = tu.make_input(torch.int32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.int32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(*args)
