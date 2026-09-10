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

# ``_slow_conv2d_backward`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# marker directly on the MarkGenerator so ``@pytest.mark._slow_conv2d_backward``
# and ``-m _slow_conv2d_backward`` both work.
setattr(
    pytest.mark,
    "_slow_conv2d_backward",
    MarkDecorator(
        Mark("_slow_conv2d_backward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride,
# padding, output_mask) -> (grad_input, grad_weight, grad_bias) is the im2col
# based "slow" conv2d backward (no dilation, groups always 1). ``self`` is
# (N, C_in, H, W), ``weight`` is (C_out, C_in, kH, kW) and ``grad_output`` is
# (N, C_out, H_out, W_out) with
#   H_out = (H + 2*pH - kH) // sH + 1, W_out = (W + 2*pW - kW) // sW + 1.
# The op exposes three overloads:
#   * .output_mask(grad_output, self, weight, kernel_size, stride, padding,
#     output_mask) -> tuple: selects which of the three gradients to compute;
#     masked-out entries are None. This is the shape the candidate is expected
#     to implement under the public name ``_slow_conv2d_backward``.
#   * .grad_input(grad_output, self, weight, kernel_size, stride, padding, *,
#     grad_input, grad_weight, grad_bias) -> tuple: writes into caller-provided
#     buffers and returns those same objects (alias semantics).
#   * .output_mask_out(..., output_mask, *, out0, out1, out2) -> tuple: same,
#     for the masked overload.
# All three are actually registered on CUDA, so they are called directly on both
# the reference and candidate paths (never simulated with default + copy_).
#
# Coverage follows the regular-operator spec adapted to this conv backward:
#   * shape/param levels: each (inp_shape, weight_shape, kernel_size, stride,
#     padding) tuple below is one distinct parametrized workload (the shared
#     tu.selected_shapes() set is pointwise-shaped and does not apply to a conv
#     whose input must be 4-D). They cover 1x1/2x2/3x3/3x5 kernels, stride 1 and
#     2, padding 0/1/2, symmetric and asymmetric kernel/stride/padding, output
#     sizes from 2 to 16 and channel counts up to 32; element counts stay well
#     below 1M.
#   * value ranges: tu.selected_ranges() resolved per-dtype by tu.make_input,
#     with the dtype-extreme ranges dropped because the op contracts over many
#     products (values near the dtype max overflow even the fp64 reference);
#     see _SLOW_CONV2D_VALUE_RANGES.
#   * broadcast: not applicable - conv requires C_in to match exactly between
#     input and weight (any mismatch is a negative case below).
#   * backward: the three gradients are the analytic backward of the forward
#     conv. They are cross-checked against torch's own autograd on the fp64
#     upcast forward, i.e. via an independent computation path.
#   * negative: inconsistent shapes/channels/kernel, non-4-D grad_output,
#     integer dtype and bare-int scalar params all raise in the reference (and
#     must raise in the candidate).
#   * nan/inf: deterministic propagation through the im2col GEMM.
#
# Dtype coverage: the reference op only supports floating point. A probe on the
# active device reports fp16 / fp32 / bf16 / fp64 as supported and rejects
# int8, uint8, fp8_e4m3fn, fp8_e5m2, int32 and int64 with "not implemented", so
# the 9-dtype spec grid collapses to the floating point dtypes below (an
# integer-dtype negative case is kept).
if QUICK_MODE:
    SLOW_CONV2D_BACKWARD_CASES = [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
    SLOW_CONV2D_VALUE_RANGES_CASES = list(SLOW_CONV2D_BACKWARD_CASES)
    FLOAT_DTYPES = [torch.float32]
else:
    SLOW_CONV2D_BACKWARD_CASES = [
        ((16, 4, 8, 8), (4, 4, 3, 3), (3, 3), (1, 1), (0, 0)),
        ((8, 3, 16, 16), (8, 3, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((32, 8, 8, 8), (32, 8, 2, 2), (2, 2), (2, 2), (0, 0)),
        ((32, 8, 8, 8), (32, 8, 2, 2), (2, 2), (1, 1), (1, 1)),
        ((4, 16, 4, 4), (16, 16, 1, 1), (1, 1), (1, 1), (0, 0)),
        ((4, 16, 4, 4), (16, 16, 1, 1), (1, 1), (2, 2), (0, 0)),
        ((2, 3, 9, 9), (4, 3, 3, 5), (3, 5), (1, 2), (1, 2)),
        ((2, 3, 4, 4), (5, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
    ]
    # The backward is reduction-heavy, so the value-range sweep stays on a
    # representative subset; the multi-case tests below still cover every case.
    SLOW_CONV2D_VALUE_RANGES_CASES = SLOW_CONV2D_BACKWARD_CASES[:4]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)

_FULL_MASK = (True, True, True)
_MIXED_MASKS = [(True, False, True), (False, True, True)]

# Inputs are scaled down before the value-range tensors are built. The three
# gradients are reduction-heavy (grad_input contracts over C_out*kH*kW and
# grad_weight/grad_bias over N*H_out*W_out), so fp16/bf16 implementations
# accumulate rounding noise proportional to the data magnitude. A modest scale
# keeps that noise well inside the dtype resolution tolerance
# (atol=1e-4*reduce_dim, rtol=RESOLUTION[dtype]) without hiding real
# indexing/formula bugs.
_INPUT_SCALE = 0.1

# The value-range sweep reuses tu.selected_ranges() (the spec ranges resolved
# per-dtype by tu.make_input) but drops the dtype-extreme ones: the op contracts
# over many products, so values near the dtype max overflow even the fp64
# reference. The remaining ranges still cover negative, positive, mixed and
# zero-containing inputs for every dtype.
_UNSAFE_FOR_REDUCTION = frozenset({"max", "min", "max/2", "min/2"})
_SLOW_CONV2D_VALUE_RANGES = [
    value_range
    for value_range in tu.selected_ranges()
    if not ({value_range[0], value_range[1]} & _UNSAFE_FOR_REDUCTION)
]

# Invalid configurations for the negative tests, as
# (inp_shape, weight_shape, kernel_size, stride, padding, grad_output_shape):
# channel mismatches (C_in/C_out), kernel_size disagreeing with the weight
# spatial dims, and grad_output shapes inconsistent with the conv output size.
_INVALID_SLOW_CONV2D_CASES = [
    # weight has wrong C_in vs input
    ((2, 3, 5, 5), (4, 5, 3, 3), (3, 3), (1, 1), (0, 0), (2, 4, 3, 3)),
    # weight spatial dims disagree with kernel_size -> grad_output H check fires
    ((2, 3, 5, 5), (4, 3, 4, 4), (3, 3), (1, 1), (0, 0), (2, 4, 2, 2)),
    # grad_output H_out inconsistent with (H, kernel, stride, padding)
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (2, 4, 6, 6)),
    # grad_output has wrong C_out vs weight
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (2, 5, 3, 3)),
    # grad_output is not 4-D
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (2, 4, 3)),
]

_NON_FLOAT_DTYPES = [torch.int32] if QUICK_MODE else [torch.int32, torch.int64]
_SCALAR_PARAMS = ["kernel_size", "stride", "padding"]


def _resolve_gems_op():
    """Resolve the candidate for the public ``_slow_conv2d_backward`` name.

    Resolved inside each test (never at import time) so that the process-local
    override installed by KernelGen for this run wins. The default stays None
    until ``flag_gems._slow_conv2d_backward`` is registered; resolution order is:
    (1) override, (2) the direct ``flag_gems._slow_conv2d_backward`` callable,
    (3) LookupError.
    """
    return flag_gems.testing.resolve_gems_op(
        "_slow_conv2d_backward", getattr(flag_gems, "_slow_conv2d_backward", None)
    )


def _make_inputs(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, value_range
):
    """Build (grad_output, input, weight) for one workload.

    All three tensors come from the value-range framework (scaled by
    ``_INPUT_SCALE``) so the same workload can be replayed for every range.
    """
    n_in, _, h_in, w_in = inp_shape
    out_c = weight_shape[0]
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    inp = _INPUT_SCALE * tu.make_input(dtype, inp_shape, value_range)
    weight = _INPUT_SCALE * tu.make_input(dtype, weight_shape, value_range)
    grad_output = _INPUT_SCALE * tu.make_input(
        dtype, (n_in, out_c, h_out, w_out), value_range
    )
    return inp, weight, grad_output


def _reference_output_mask(
    inp, weight, grad_output, kernel_size, stride, padding, mask
):
    """High-precision (fp64 upcast) reference computed with torch.ops.aten."""
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    return torch.ops.aten._slow_conv2d_backward.output_mask(
        ref_grad_output,
        ref_inp,
        ref_weight,
        kernel_size,
        stride,
        padding,
        mask,
    )


def _reduction_dims(inp_shape, weight_shape, stride, padding):
    """Contraction sizes used to scale atol for each gradient compare."""
    n_in, _, h_in, w_in = inp_shape
    out_c, _, k_h, k_w = weight_shape
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    in_reduce_dim = out_c * k_h * k_w  # grad_input contracts over C_out x kH x kW
    out_reduce_dim = (
        n_in * h_out * w_out
    )  # grad_weight/bias contract over N x H_out x W_out
    return in_reduce_dim, out_reduce_dim


def _assert_grads_close(
    res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=False
):
    """Compare the (grad_input, grad_weight, grad_bias) tuple element-wise.

    Masked-out entries must be ``None`` on both sides; computed entries are
    compared with the dtype-resolution tolerance scaled by the contraction size.
    """
    res_in_grad, res_weight_grad, res_bias_grad = res
    ref_in_grad, ref_weight_grad, ref_bias_grad = ref
    if ref_in_grad is None:
        assert res_in_grad is None
    else:
        utils.gems_assert_close(
            res_in_grad,
            ref_in_grad,
            dtype,
            reduce_dim=in_reduce_dim,
            equal_nan=equal_nan,
        )
    if ref_weight_grad is None:
        assert res_weight_grad is None
    else:
        utils.gems_assert_close(
            res_weight_grad,
            ref_weight_grad,
            dtype,
            reduce_dim=out_reduce_dim,
            equal_nan=equal_nan,
        )
    if ref_bias_grad is None:
        assert res_bias_grad is None
    else:
        utils.gems_assert_close(
            res_bias_grad,
            ref_bias_grad,
            dtype,
            reduce_dim=out_reduce_dim,
            equal_nan=equal_nan,
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_VALUE_RANGES_CASES)
@pytest.mark.parametrize("value_range", _SLOW_CONV2D_VALUE_RANGES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_value_ranges(case, value_range, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, value_range
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_full(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_only(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (True, False, False)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_weight_only(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (False, True, False)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_bias_only(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (False, False, True)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES[:2])
@pytest.mark.parametrize("mask", _MIXED_MASKS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_mixed_mask(case, mask, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_out(case, dtype):
    """Exercise the real ``.grad_input`` overload on both paths.

    The overload writes into caller-provided buffers and returns those same
    tensor objects (alias semantics). Buffers are garbage-prefilled so the
    overload must fully overwrite them.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    ref_grad_input = torch.full_like(inp, 7.0)
    ref_grad_weight = torch.full_like(weight, 7.0)
    ref_grad_bias = torch.full(
        (weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device
    )
    ref_ret = torch.ops.aten._slow_conv2d_backward.grad_input(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        grad_input=ref_grad_input,
        grad_weight=ref_grad_weight,
        grad_bias=ref_grad_bias,
    )
    assert ref_ret[0] is ref_grad_input
    assert ref_ret[1] is ref_grad_weight
    assert ref_ret[2] is ref_grad_bias

    res_grad_input = torch.full_like(inp, 7.0)
    res_grad_weight = torch.full_like(weight, 7.0)
    res_grad_bias = torch.full(
        (weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device
    )
    res = _resolve_gems_op()(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        grad_input=res_grad_input,
        grad_weight=res_grad_weight,
        grad_bias=res_grad_bias,
    )
    assert res[0] is res_grad_input
    assert res[1] is res_grad_weight
    assert res[2] is res_grad_bias

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)
    _assert_grads_close(ref_ret, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_out(case, dtype):
    """Exercise the real ``.output_mask_out`` overload on both paths."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    ref_out0 = torch.full_like(inp, 7.0)
    ref_out1 = torch.full_like(weight, 7.0)
    ref_out2 = torch.full((weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device)
    ref_ret = torch.ops.aten._slow_conv2d_backward.output_mask_out(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        out0=ref_out0,
        out1=ref_out1,
        out2=ref_out2,
    )
    assert ref_ret[0] is ref_out0
    assert ref_ret[1] is ref_out1
    assert ref_ret[2] is ref_out2

    res_out0 = torch.full_like(inp, 7.0)
    res_out1 = torch.full_like(weight, 7.0)
    res_out2 = torch.full((weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device)
    res = _resolve_gems_op()(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        out0=res_out0,
        out1=res_out1,
        out2=res_out2,
    )
    assert res[0] is res_out0
    assert res[1] is res_out1
    assert res[2] is res_out2

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)
    _assert_grads_close(ref_ret, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES[:3])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_backward(case, dtype):
    """Cross-check the gradients with torch's own autograd on the forward op."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    bias = _INPUT_SCALE * tu.make_input(dtype, (weight_shape[0],), ["-1", "1"])

    # Differentiate sum(forward * grad_output) w.r.t. (input, weight, bias)
    # through autograd on the fp64 upcast reference and compare against the
    # candidate's three gradients. This validates the candidate against the true
    # gradient through an independent computation path.
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_weight = utils.to_reference(weight, True).requires_grad_(True)
    ref_bias = utils.to_reference(bias, True).requires_grad_(True)
    ref_grad_output = utils.to_reference(grad_output, True)

    fwd = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    )
    ref = torch.autograd.grad(
        (fwd * ref_grad_output).sum(),
        (ref_inp, ref_weight, ref_bias),
        allow_unused=True,
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_nan_inf(dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = _INPUT_SCALE * tu.make_input(dtype, (2, 3, 5, 5), ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(dtype, (2, 3, 3, 3), ["-1", "1"])
    grad_output = _INPUT_SCALE * tu.make_input(dtype, (2, 2, 5, 5), ["-1", "1"])

    # Poison a few entries with nan / inf / -inf. The fp64-upcast reference sees
    # exactly the same values, so the special values propagate identically
    # through the im2col products on both paths; equal_nan tolerates the nan
    # entries produced by nan products and inf + (-inf) accumulation.
    inp[0, 0, 2, 2] = float("nan")
    inp[1, 2, 1, 4] = float("inf")
    weight[1, 1, 0, 0] = float("-inf")
    grad_output[0, 1, 3, 3] = float("nan")
    grad_output[1, 0, 0, 0] = float("inf")

    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    res = _resolve_gems_op()(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        (2, 3, 5, 5), (2, 3, 3, 3), stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=True)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", _INVALID_SLOW_CONV2D_CASES)
def test__slow_conv2d_backward_negative_invalid_config(case):
    inp_shape, weight_shape, kernel_size, stride, padding, grad_output_shape = case
    inp = _INPUT_SCALE * tu.make_input(torch.float32, inp_shape, ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, weight_shape, ["-1", "1"])
    grad_output = _INPUT_SCALE * tu.make_input(
        torch.float32, grad_output_shape, ["-1", "1"]
    )

    # The reference rejects the inconsistent configuration...
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
        )

    # ...and so must the candidate. LookupError is tolerated only when no
    # candidate has been injected for this run.
    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(
            grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("dtype", _NON_FLOAT_DTYPES)
def test__slow_conv2d_backward_negative_non_float_dtype(dtype):
    # The im2col backward only supports floating point inputs.
    inp = tu.make_input(dtype, (2, 3, 5, 5), ["0", "1"])
    weight = tu.make_input(dtype, (4, 3, 3, 3), ["0", "1"])
    grad_output = tu.make_input(dtype, (2, 4, 3, 3), ["0", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK)


@pytest.mark._slow_conv2d_backward
def test__slow_conv2d_backward_negative_non_4d_grad_output():
    # grad_output must be (N, C_out, H_out, W_out).
    inp = _INPUT_SCALE * tu.make_input(torch.float32, (2, 3, 5, 5), ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, (4, 3, 3, 3), ["-1", "1"])
    grad_output = _INPUT_SCALE * tu.make_input(torch.float32, (2, 4, 3), ["-1", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("scalar_param", _SCALAR_PARAMS)
def test__slow_conv2d_backward_negative_scalar_param(scalar_param):
    # kernel_size/stride/padding are SymInt[2]: a bare scalar int does not match
    # the schema and raises.
    inp_shape, weight_shape, kernel_size, stride, padding = SLOW_CONV2D_BACKWARD_CASES[
        0
    ]
    inp = _INPUT_SCALE * tu.make_input(torch.float32, inp_shape, ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, weight_shape, ["-1", "1"])
    n_in, _, h_in, w_in = inp_shape
    out_c = weight_shape[0]
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    grad_output = _INPUT_SCALE * tu.make_input(
        torch.float32, (n_in, out_c, h_out, w_out), ["-1", "1"]
    )

    args = [kernel_size, stride, padding]
    args[_SCALAR_PARAMS.index(scalar_param)] = 3

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, args[0], args[1], args[2], _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError, LookupError)):
        _resolve_gems_op()(
            grad_output, inp, weight, args[0], args[1], args[2], _FULL_MASK
        )
