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

# aten::slow_conv_dilated3d(Tensor self, Tensor weight, SymInt[3] kernel_size,
# Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0],
# SymInt[3] dilation=[1, 1, 1]) -> Tensor is the im2col based "slow" conv3d with
# dilation support (groups always 1). ``self`` is (N, C_in, D, H, W), ``weight``
# is (C_out, C_in, kD, kH, kW) and ``kernel_size`` must match the weight spatial
# dims. The output is (N, C_out, D_out, H_out, W_out) with
#   D_out = (D + 2*pD - dil_d*(kD - 1) - 1) // sD + 1
# and likewise for H and W. Each (input, weight, kernel_size, stride, padding,
# dilation) tuple below is one distinct parametrized workload: they cover
# 1x1x1/2x2x2/3x3x3 kernels, stride 1/2, padding 0/1/2, dilation 1/2, and
# asymmetric strides/paddings, with and without bias. Element counts stay well
# below 1M so the correctness run stays fast.
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
    # fp64 reference by up to ~8e-3 (fp16) and ~1.1e-1 (bf16) on the larger
    # 3D reductions (up to C_in*kD*kH*kW = 108 terms). Measure the deviation
    # over multiple seeds and shapes: fp16 -> 2e-2 and bf16 -> 2e-1 give a
    # ~2x margin; fp32 with TF32 disabled (set at the top of each test) stays
    # at ~9e-6, comfortably inside the default 1e-4.
    if dtype == torch.bfloat16:
        atol = 2e-1
    elif dtype == torch.float16:
        atol = 2e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


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
