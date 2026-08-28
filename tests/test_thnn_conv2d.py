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
from .conftest import QUICK_MODE  # noqa: E402

# aten::thnn_conv2d(Tensor self, Tensor weight, SymInt[2] kernel_size,
# Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0]) -> Tensor
# is the im2col based "slow" conv2d (no dilation, groups always 1). ``self`` is
# (N, C_in, H, W), ``weight`` is (C_out, C_in, kH, kW) and ``kernel_size`` must
# match the weight spatial dims. The output is (N, C_out, H_out, W_out) with
#   H_out = (H + 2*pH - kH) // sH + 1, W_out = (W + 2*pW - kW) // sW + 1.
# Each (input, weight, kernel_size, stride, padding) tuple below is one distinct
# parametrized workload: they cover 1x1/3x3/3x5/5x5 kernels, stride 1 and 2,
# padding 0/1/2, small outputs and channel counts up to 32. Element counts stay
# well below 1M so the correctness run stays fast.
if QUICK_MODE:
    THNN_CONV2D_CASES = [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ]
    FLOAT_DTYPES = [torch.float32]
    BIASES = [True]
else:
    THNN_CONV2D_CASES = [
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


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.thnn_conv2d is registered; resolution order is:
    # (1) override, (2) the direct flag_gems.thnn_conv2d callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "thnn_conv2d", getattr(flag_gems, "thnn_conv2d", None)
    )


def _resolve_gems_op_out():
    # The .out overload is a distinct public operator name ("thnn_conv2d.out");
    # the direct flag_gems attribute stays None until the op is registered.
    return flag_gems.testing.resolve_gems_op(
        "thnn_conv2d.out", getattr(flag_gems, "thnn_conv2d_out", None)
    )


def _make_conv_inputs(inp_shape, weight_shape, with_bias, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    if with_bias:
        bias = torch.randn(weight_shape[0], dtype=dtype, device=flag_gems.device)
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


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", THNN_CONV2D_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_thnn_conv2d(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    # The reference op runs cuBLAS for the im2col GEMM; keep TF32 off so the
    # fp32 comparison stays at the standard 1e-4 tolerance (with TF32 on the
    # native op itself deviates from the fp64 reference by ~1.6e-2).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    ref_out = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(inp, weight, kernel_size, bias_t, stride, padding)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.thnn_conv2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", THNN_CONV2D_CASES
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_thnn_conv2d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    # The .out overload shares the .default compute. Both the ATen reference and
    # the candidate must write into the caller-provided buffer and return it.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias_t, True)

    # Compute the expected output shape via the .default reference, then hand
    # both the .out reference and the candidate equally-shaped out buffers (the
    # reference out must match the fp64 reference dtype).
    expected = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    ref_out_t = torch.empty(expected.shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_ret = torch.ops.aten.thnn_conv2d.out(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, out=ref_out_t
    )

    out_t = torch.empty(expected.shape, dtype=dtype, device=flag_gems.device)
    gems_op = _resolve_gems_op_out()
    res_ret = gems_op(inp, weight, kernel_size, bias_t, stride, padding, out=out_t)

    # Alias semantics: the .out overload returns the caller-provided buffer.
    assert ref_ret is ref_out_t
    assert res_ret is out_t
    _assert_close(res_ret, ref_ret, dtype)
    _assert_close(out_t, ref_out_t, dtype)
