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
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402
from .conftest import QUICK_MODE  # noqa: E402

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
# The ``output_mask`` selects which of the three gradients to compute; the
# masked-out entries are None. The .grad_input and .output_mask_out overloads
# write into caller-provided buffers and return the same tensor objects (alias
# semantics). Each (inp_shape, weight_shape, kernel_size, stride, padding) tuple
# below is one distinct parametrized workload: they cover 1x1/2x2/3x3/3x5
# kernels, stride 1 and 2, padding 0/1/2, symmetric and asymmetric
# kernel/stride/padding and output sizes from 2 to 16, with channel counts up
# to 32. Element counts stay well below 1M so the correctness run stays fast.
if QUICK_MODE:
    SLOW_CONV2D_BACKWARD_CASES = [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
    SLOW_CONV2D_VALUE_RANGES_CASES = [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
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
    SLOW_CONV2D_VALUE_RANGES_CASES = [
        ((16, 4, 8, 8), (4, 4, 3, 3), (3, 3), (1, 1), (0, 0)),
        ((4, 16, 4, 4), (16, 16, 1, 1), (1, 1), (2, 2), (0, 0)),
        ((8, 3, 16, 16), (8, 3, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)

_FULL_MASK = (True, True, True)

# Inputs are scaled down before the value-range tensors are built. The three
# gradients are reduction-heavy (each output contracts over C_out*kH*kW for
# grad_input and N*H_out*W_out for grad_weight/grad_bias), so fp16/bf16
# implementations accumulate rounding noise proportional to the data magnitude.
# A modest scale keeps that noise well inside the dtype resolution tolerance
# (atol=1e-4*reduce_dim, rtol=RESOLUTION[dtype]) without changing the relative
# precision of the computation, while still catching real indexing/formula bugs.
# Value ranges are the shared regular-operator ranges (``tu.selected_ranges()``
# minus the extreme max/min ones, whose magnitudes would push fp16/bf16
# reduction noise past the tolerance); all of them are safe at this scale.
_INPUT_SCALE = 0.1
_SLOW_CONV2D_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
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


def _make_inputs(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, value_range
):
    """Build (input, weight, grad_output) for a single conv2d workload whose
    values come from the value-range framework (scaled by _INPUT_SCALE)."""
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


def _gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._slow_conv2d_backward is registered; resolution order is:
    # (1) override, (2) the direct flag_gems._slow_conv2d_backward callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_slow_conv2d_backward", getattr(flag_gems, "_slow_conv2d_backward", None)
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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK)

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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, mask)

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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, mask)

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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, mask)

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_out(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    grad_input = torch.zeros_like(inp)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros(weight_shape[0], dtype=dtype, device=flag_gems.device)

    # .grad_input overload: writes into the provided buffers and returns them.
    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        grad_input=grad_input,
        grad_weight=grad_weight,
        grad_bias=grad_bias,
    )

    # Alias semantics: the returned tuple references the same tensor objects.
    assert res[0] is grad_input
    assert res[1] is grad_weight
    assert res[2] is grad_bias
    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_out(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    out0 = torch.zeros_like(inp)
    out1 = torch.zeros_like(weight)
    out2 = torch.zeros(weight_shape[0], dtype=dtype, device=flag_gems.device)

    # .output_mask_out overload: writes into out0/out1/out2 and returns them.
    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        out0=out0,
        out1=out1,
        out2=out2,
    )

    assert res[0] is out0
    assert res[1] is out1
    assert res[2] is out2
    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_backward(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    bias = _INPUT_SCALE * tu.make_input(dtype, (weight_shape[0],), ["-1", "1"])

    # The backward op is the analytic gradient of the forward conv. Differentiate
    # sum(forward * grad_output) w.r.t. (input, weight, bias) through autograd on
    # the fp64 upcast reference and compare against the candidate's outputs. This
    # cross-checks the candidate against the true gradient through an independent
    # computation path (torch's own autograd instead of the same backward kernel).
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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK)

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
    # exactly the same values, so nan/inf propagate identically through the im2col
    # products on both paths; equal_nan=True tolerates the resulting nan entries.
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

    gems_op = _gems_op()
    res = gems_op(grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK)

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        (2, 3, 5, 5), (2, 3, 3, 3), stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=True)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", _INVALID_SLOW_CONV2D_CASES)
def test__slow_conv2d_backward_negative(case):
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
        _gems_op()(grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK)
