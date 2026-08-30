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

# aten::abs(Tensor self) -> Tensor computes the elementwise absolute value:
# negatives flip sign, non-negatives are the identity, and |INT_MIN| == INT_MIN
# (PyTorch defines no wrap-around). The op is exact for every storage dtype, so
# the value-range comparisons below are bit-for-bit on the int/bool path and
# well inside the default float tolerance (equal_nan=True covers the nan/inf
# cases). The .default overload is resolved through its public name "abs"
# (KernelGen's override_gems_op("abs", ...) wins over the direct callable), the
# in-place variant through "abs_", and the .out overload through "abs.out"
# whose default implementation is the adapter below; KernelGen may override
# "abs.out" with a real out-kernel.
#
# Dtype coverage: the op is defined for every storage dtype; the value-range
# tests run over the full float (fp16/fp32/bf16/fp64), int (int16/int32/int64)
# and bool families.
_ABS_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_ABS_INT_DTYPES = utils.ALL_INT_DTYPES
_ABS_DTYPES = _ABS_FLOAT_DTYPES + _ABS_INT_DTYPES + utils.BOOL_TYPES

# Shapes that exercise 0-dim scalars, degenerate/empty tensors and
# non-contiguous strides (the pointwise kernel must honor the input strides).
_ABS_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_ABS_NONCONTIG_SHAPES = [(17, 33), (5, 7, 9)]

# Backward shapes stay small (the autograd graph is built on the CPU reference
# and the analytic comparison below is elementwise).
_ABS_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.abs callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("abs", flag_gems.abs)


def _resolve_gems_op_inplace():
    return flag_gems.testing.resolve_gems_op("abs_", flag_gems.abs_)


def _abs_out_adapter(self, *, out):
    # Default implementation of the ".out" overload: run the direct abs kernel
    # and copy the result into the caller's out buffer. KernelGen's override of
    # "abs.out" replaces this adapter with a real out-kernel.
    out.copy_(flag_gems.testing.resolve_gems_op("abs", flag_gems.abs)(self))
    return out


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op("abs.out", _abs_out_adapter)


@pytest.mark.abs
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_FLOAT_DTYPES)
def test_abs_float_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_INT_DTYPES + utils.BOOL_TYPES)
def test_abs_int_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    # int/bool abs is exact: assert_result_close uses an atol=0/rtol=0 path.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", _ABS_FLOAT_DTYPES)
def test_abs_nan_inf(dtype):
    # inf/-inf -> +inf, nan -> nan, -0.0 -> 0.0. 1e30/-1e30 also cover the
    # overflow-to-inf path in fp16/bf16; equal_nan=True tolerates nan outputs.
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", _ABS_INT_DTYPES)
def test_abs_int_min_stays(dtype):
    # |INT_MIN| == INT_MIN in PyTorch (no wrap-around); pin this contract.
    min_val = torch.iinfo(dtype).min
    inp = torch.tensor(
        [min_val, min_val + 1, 0, 1, -1], dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", _ABS_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", _ABS_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_noncontiguous(shape, dtype):
    # transposed views have non-unit strides; the kernel must honor them.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", _ABS_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.abs(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    # d|x|/dx == sign(x) (torch defines sign(0) == 0), so the reference
    # gradient must match the analytic value; this validates the reference
    # autograd path itself.
    expected_in_grad = torch.sign(ref_inp) * ref_grad
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward output must match the reference...
    res_out = _resolve_gems_op()(inp)
    tu.assert_result_close(res_out, ref_out)

    # ...and, if the candidate kernel advertises autograd support (the current
    # direct kernel does not: res_out.requires_grad is False), its gradient
    # must match the analytic value too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark.abs_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs__value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.abs_(ref_inp)
    res_out = _resolve_gems_op_inplace()(inp)

    # In-place semantics: the call returns the mutated input tensor itself.
    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.abs_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_out(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.abs.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs_negative
def test_abs_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.abs(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
