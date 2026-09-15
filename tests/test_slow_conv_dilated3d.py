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
# Dtype coverage: the CUDA kernel implements the floating dtypes fp16 / fp32 /
# bf16 / fp64 -- int8, uint8, float8_e4m3fn, float8_e5m2, int32 and int64 all
# raise `"slow_conv_dilated<>" not implemented for '<Int/Float8...>'`. The
# correctness grid therefore uses utils.ALL_FLOAT_DTYPES; integer/fp8 rejection
# is covered by the negative dtype test below instead of being silently
# dropped.
#
# Rank probe: the schema accepts both a batched 5-D input (N, C_in, D, H, W)
# and an unbatched 4-D one (C_in, D, H, W), so the 4-D rank IS exercised (see
# test_slow_conv_dilated3d_unbatched). aten does not support the unbatched 4-D
# route together with a bias on this build -- measured, some configurations
# raise
#   IndexError: select(): index 4 out of range for tensor of size [4, 4, 4]
# and the rest return non-deterministic results (repeated identical calls
# disagree by up to 1.1e38, incl. inf/NaN) that differ from F.conv3d by O(1).
# The 4-D rank is therefore covered without a bias: there the op is
# deterministic (0.0 across repeated calls) and matches F.conv3d to 0.0 in
# fp64. The bias path is covered by the batched grid, and the 6-D rejection
# test covers the rank check's negative side.
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
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
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


def _unbatched(case):
    # (N, C_in, D, H, W) -> the unbatched (C_in, D, H, W) form of the same
    # operator configuration; everything else is unchanged.
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    return (inp_shape[1:], weight_shape, kernel_size, stride, padding, dilation)


# The 4-D unbatched route, on the same three representative configurations the
# value-range test uses (3x3x3/pad1, 1x1x1 pure-GEMM, pad2) so the rank is
# covered without duplicating the whole batched grid. Bias is deliberately
# excluded: aten does not support 4-D + bias (see the rank-probe note above),
# so it cannot serve as the oracle.
_UNBATCHED_CASES = [
    _unbatched(SLOW_CONV_DILATED3D_CASES[i]) for i in ([0] if QUICK_MODE else [0, 5, 4])
]

# Extreme workloads retain their declared bounds. Their oracle uses the original
# dtype because upcasting changes intermediate overflow and NaN propagation.
_VALUE_RANGE_CASES = (
    SLOW_CONV_DILATED3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_DILATED3D_CASES[0],  # 3x3x3 kernel, padding 1, dilation 1
        SLOW_CONV_DILATED3D_CASES[5],  # 1x1x1 kernel (pure GEMM path)
        SLOW_CONV_DILATED3D_CASES[3],  # dilation 2
    ]
)
_VALUE_RANGES = tu.selected_ranges()

# The aten op carries its own autograd (SlowConvDilated3DBackward0), so
# backward is exercised directly. fp16/bf16 are included: the native op
# accumulates its gradients in the input dtype and the dtype-scaled _assert_close
# tolerances below (fp16 2e-2, bf16 2e-1, plus the per-dtype rtol) cover that
# rounding, as they do for the forward.
_BACKWARD_CASES = (
    SLOW_CONV_DILATED3D_CASES[:1]
    if QUICK_MODE
    else [
        SLOW_CONV_DILATED3D_CASES[0],  # padding 1, dilation 1
        SLOW_CONV_DILATED3D_CASES[1],  # no padding
        SLOW_CONV_DILATED3D_CASES[3],  # dilation 2
    ]
)
_BACKWARD_DTYPES = (
    [torch.float32]
    if QUICK_MODE
    else [torch.float16, torch.float32, torch.bfloat16, torch.float64]
)


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "slow_conv_dilated3d", getattr(flag_gems, "slow_conv_dilated3d", None)
    )


def _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation):
    """Batched (N, C_in, D, H, W) or unbatched (C_in, D, H, W) input x
    (C_out, C_in, kD, kH, kW) weight -> the matching (N, C_out, ...) or
    (C_out, ...) output shape."""

    def _out_size(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    # A 4-D input is the unbatched route: the spatial dims start one offset
    # earlier and the output drops the batch dim too.
    if len(inp_shape) == 4:
        spatial_offset, lead = 1, (weight_shape[0],)
    else:
        spatial_offset, lead = 2, (inp_shape[0], weight_shape[0])

    spatial = tuple(
        _out_size(
            inp_shape[spatial_offset + i],
            weight_shape[2 + i],
            stride[i],
            padding[i],
            dilation[i],
        )
        for i in range(3)
    )
    return lead + spatial


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
    # Finite random workloads use an fp64 reference. These absolute bounds
    # supplement the shared relative tolerance for the original dtype.
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
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    gems_op = _resolve_gems_op()
    res_out = gems_op(inp, weight, kernel_size, bias_t, stride, padding, dilation)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _UNBATCHED_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_slow_conv_dilated3d_unbatched(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype
):
    # The unbatched 4-D route (C_in, D, H, W) must produce the same result as
    # the batched one -- the operator's rank dimension is part of the spec's
    # shape coverage, so it is checked rather than skipped. No bias: aten does
    # not support 4-D + bias, so it cannot be used as the oracle there.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, False, dtype)
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, None, stride, padding, dilation
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    # The output must also drop the batch dim, not silently keep a leading 1.
    assert tuple(res_out.shape) == _conv_output_shape(
        inp_shape, weight_shape, stride, padding, dilation
    )
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
    ref_inp = tu.to_reference(inp, not tu.is_extreme_range(value_range))
    ref_weight = tu.to_reference(weight, not tu.is_extreme_range(value_range))
    ref_bias = tu.to_reference(bias_t, not tu.is_extreme_range(value_range))

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)


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
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

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

    out = torch.empty(ref_full.shape, dtype=dtype, device=flag_gems.device)
    res_ret = _resolve_gems_op()(
        inp, weight, kernel_size, bias_t, stride, padding, dilation, out=out
    )
    assert res_ret is out

    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_dilated3d_backward
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES))
def test_slow_conv_dilated3d_backward(case, dtype):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    out_shape = _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, inp_shape, ["-1", "1"]).requires_grad_()
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"]).requires_grad_()
    bias = tu.make_input(dtype, (weight_shape[0],), ["-1", "1"]).requires_grad_()
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    # Reference graph on the fp64-upcast inputs.
    ref_inp = tu.to_reference(inp, True).requires_grad_()
    ref_weight = tu.to_reference(weight, True).requires_grad_()
    ref_bias = tu.to_reference(bias, True).requires_grad_()
    ref_grad_out = tu.to_reference(grad_out, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_gi, ref_gw, ref_gb = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )

    # Self-check: the low-level op's autograd must match the standard
    # F.conv3d backward (same math, im2col vs direct formulation).
    f_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        ref_bias,
        stride=stride,
        padding=padding,
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
        inp, weight, kernel_size, bias, stride, padding, dilation
    )
    _assert_close(res_out, ref_out.to(dtype), dtype)

    # ...and, if the candidate kernel is autograd-aware, its gradients must
    # match the reference gradients too.
    assert res_out.requires_grad
    res_gi, res_gw, res_gb = torch.autograd.grad(
        res_out, (inp, weight, bias), grad_outputs=grad_out
    )
    _assert_close(res_gi, ref_gi.to(dtype), dtype)
    _assert_close(res_gw, ref_gw.to(dtype), dtype)
    _assert_close(res_gb, ref_gb.to(dtype), dtype)


@pytest.mark.slow_conv_dilated3d_nan_inf
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "bias"])
def test_slow_conv_dilated3d_nan_inf(dtype, scenario, special_arg):
    # Exact finite backgrounds isolate special-value propagation in each operand.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    inp = torch.ones((1, 1, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((5, 1, 2, 2, 2), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((5,), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "bias": bias}[special_arg]
    target.flatten()[: specials.numel()] = specials
    kernel_size = (2, 2, 2)

    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    ).to(dtype)

    res_out = _resolve_gems_op()(
        inp, weight, kernel_size, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    )

    tu.assert_result_close(res_out, ref_out)


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
