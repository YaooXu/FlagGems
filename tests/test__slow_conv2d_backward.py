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
# padding, output_mask) returns (grad_input, grad_weight, grad_bias). ``self`` is
# (N, C_in, H, W), ``weight`` is (C_out, C_in, kH, kW) and ``grad_output`` is
# (N, C_out, H_out, W_out) with H_out = (H + 2*pH - kH) // sH + 1. The
# ``output_mask`` selects which of the three gradients to compute; the masked-out
# entries are None. The .grad_input and .output_mask_out overloads write into
# caller-provided buffers and return the same tensor objects (alias semantics).
# Each (shape, weight_shape) tuple below is one distinct parametrized workload:
# they cover 1x1/2x2/3x3 kernels, stride 1 and 2, padding 0/1, output sizes from
# 2 to 16 and channel counts up to 32. Element counts stay well below 1M so the
# correctness run stays fast.
if QUICK_MODE:
    SHAPE_SLOW_CONV2D = [
        ((1, 2, 5, 5), (2, 2, 3, 3)),
    ]
    STRIDES = [1]
    PADDINGS = [1]
    FLOAT_DTYPES = [torch.float32]
else:
    SHAPE_SLOW_CONV2D = [
        ((16, 4, 8, 8), (4, 4, 3, 3)),
        ((8, 3, 16, 16), (8, 3, 3, 3)),
        ((32, 8, 8, 8), (32, 8, 2, 2)),
        ((4, 16, 4, 4), (16, 16, 1, 1)),
    ]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES  # fp16, fp32, bf16, (+fp64)

_FULL_MASK = (True, True, True)

# Inputs are scaled down before the fp64 upcast reference is computed. The three
# gradients are reduction-heavy (each output contracts over C_out*kH*kW for
# grad_input and N*H_out*W_out for grad_weight/grad_bias), so fp16/bf16
# implementations accumulate rounding noise proportional to the data magnitude.
# A modest scale keeps that noise well inside the dtype resolution tolerance
# (atol=1e-4*reduce_dim, rtol=RESOLUTION[dtype]) without changing the relative
# precision of the computation, while still catching real indexing/formula bugs.
_INPUT_SCALE = 0.1


def _make_inputs(shape, weight_shape, stride, padding, dtype):
    """Build (input, weight, grad_output) for a single conv2d workload."""
    n_in, _, h_in, w_in = shape
    out_c, _, k_h, k_w = weight_shape
    h_out = (h_in + 2 * padding - k_h) // stride + 1
    w_out = (w_in + 2 * padding - k_w) // stride + 1
    inp = _INPUT_SCALE * torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = _INPUT_SCALE * torch.randn(
        weight_shape, dtype=dtype, device=flag_gems.device
    )
    grad_output = _INPUT_SCALE * torch.randn(
        (n_in, out_c, h_out, w_out), dtype=dtype, device=flag_gems.device
    )
    return inp, weight, grad_output


def _reference_output_mask(inp, weight, grad_output, stride, padding, mask):
    """High-precision (fp64 upcast) reference computed with torch.ops.aten."""
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    k_h, k_w = weight.shape[2], weight.shape[3]
    return torch.ops.aten._slow_conv2d_backward.output_mask(
        ref_grad_output,
        ref_inp,
        ref_weight,
        (k_h, k_w),
        (stride, stride),
        (padding, padding),
        mask,
    )


def _gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._slow_conv2d_backward is registered; resolution order is:
    # (1) override, (2) the direct flag_gems._slow_conv2d_backward callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_slow_conv2d_backward", getattr(flag_gems, "_slow_conv2d_backward", None)
    )


def _reduction_dims(shape, weight_shape, stride, padding):
    """Contraction sizes used to scale atol for each gradient compare."""
    n_in, _, h_in, w_in = shape
    out_c, _, k_h, k_w = weight_shape
    h_out = (h_in + 2 * padding - k_h) // stride + 1
    w_out = (w_in + 2 * padding - k_w) // stride + 1
    in_reduce_dim = out_c * k_h * k_w  # grad_input contracts over C_out x kH x kW
    out_reduce_dim = (
        n_in * h_out * w_out
    )  # grad_weight/bias contract over N x H_out x W_out
    return in_reduce_dim, out_reduce_dim


def _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype):
    res_in_grad, res_weight_grad, res_bias_grad = res
    ref_in_grad, ref_weight_grad, ref_bias_grad = ref
    if ref_in_grad is None:
        assert res_in_grad is None
    else:
        utils.gems_assert_close(
            res_in_grad, ref_in_grad, dtype, reduce_dim=in_reduce_dim
        )
    if ref_weight_grad is None:
        assert res_weight_grad is None
    else:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=out_reduce_dim
        )
    if ref_bias_grad is None:
        assert res_bias_grad is None
    else:
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=out_reduce_dim
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward(shape, weight_shape, stride, padding, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, _FULL_MASK)

    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        _FULL_MASK,
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_only(
    shape, weight_shape, stride, padding, dtype
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    mask = (True, False, False)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, mask)

    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        mask,
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_weight_only(
    shape, weight_shape, stride, padding, dtype
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    mask = (False, True, False)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, mask)

    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        mask,
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_bias_only(
    shape, weight_shape, stride, padding, dtype
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    mask = (False, False, True)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, mask)

    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        mask,
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_out(
    shape, weight_shape, stride, padding, dtype
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, _FULL_MASK)

    grad_input = torch.zeros_like(inp)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros(weight_shape[0], dtype=dtype, device=flag_gems.device)

    # .grad_input overload: writes into the provided buffers and returns them.
    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        grad_input=grad_input,
        grad_weight=grad_weight,
        grad_bias=grad_bias,
    )

    # Alias semantics: the returned tuple references the same tensor objects.
    assert res[0] is grad_input
    assert res[1] is grad_weight
    assert res[2] is grad_bias
    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("shape, weight_shape", SHAPE_SLOW_CONV2D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_out(
    shape, weight_shape, stride, padding, dtype
):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, grad_output = _make_inputs(shape, weight_shape, stride, padding, dtype)
    ref = _reference_output_mask(inp, weight, grad_output, stride, padding, _FULL_MASK)

    out0 = torch.zeros_like(inp)
    out1 = torch.zeros_like(weight)
    out2 = torch.zeros(weight_shape[0], dtype=dtype, device=flag_gems.device)

    # .output_mask_out overload: writes into out0/out1/out2 and returns them.
    gems_op = _gems_op()
    res = gems_op(
        grad_output,
        inp,
        weight,
        (weight_shape[2], weight_shape[3]),
        (stride, stride),
        (padding, padding),
        _FULL_MASK,
        out0=out0,
        out1=out1,
        out2=out2,
    )

    assert res[0] is out0
    assert res[1] is out1
    assert res[2] is out2
    in_reduce_dim, out_reduce_dim = _reduction_dims(
        shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)
