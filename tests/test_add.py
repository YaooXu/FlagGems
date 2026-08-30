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

# aten::add(Tensor self, Tensor other, *, Scalar alpha) computes the elementwise
# self + other * alpha (the .Scalar overload takes a scalar ``other``; the
# all-scalar form returns a 0-dim tensor). The op is exact for every storage
# dtype: int arithmetic wraps like two's-complement on the reference and bool
# behaves as logical OR, so the value-range comparisons below are bit-for-bit on
# the int/bool path and well inside the default float tolerance
# (equal_nan=True covers the inf + (-inf) = nan cases). The .default overload is
# resolved through its public name "add" (KernelGen's override_gems_op("add", ...)
# wins over the direct callable), the in-place variant through "add_", and the
# .out overload through "add.out" whose default implementation is the adapter
# below; KernelGen may override "add.out" with a real out-kernel.
#
# Dtype coverage: the op is defined for every storage dtype; the value-range
# tests run over the full float (fp16/fp32/bf16/fp64), int (int16/int32/int64)
# and bool families, plus complex (complex64/complex128) and the vendor-gated
# complex32 case (ascend/tsingmicro do not implement complex32).
_ADD_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_ADD_INT_DTYPES = utils.ALL_INT_DTYPES
_ADD_DTYPES = _ADD_FLOAT_DTYPES + _ADD_INT_DTYPES + utils.BOOL_TYPES
_ADD_COMPLEX_DTYPES = [torch.complex64, torch.complex128]
_ADD_COMPLEX32_DTYPES = [d for d in utils.COMPLEX_DTYPES if d == torch.complex32]

# Broadcast pairs cover the spec's canonical set in both operand orders plus a
# couple of higher-rank right-operand reductions.
_ADD_BROADCAST_PAIRS = [
    ((2, 3, 5), (5,)),
    ((2, 3, 5), (2, 1, 5)),
    ((2, 3, 5), (1, 3, 1)),
    ((5,), (2, 3, 5)),
    ((2, 1, 5), (2, 3, 5)),
    ((1, 3, 1), (2, 3, 5)),
]

# In-place add_ requires the self tensor to be the broadcast target, so only
# the pairs whose first operand is the larger shape are valid here.
_ADD_INPLACE_BROADCAST_PAIRS = _ADD_BROADCAST_PAIRS[:3]

# Shapes that exercise 0-dim scalars, degenerate/empty tensors and
# non-contiguous strides (the pointwise kernel must honor the input strides).
_ADD_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_ADD_NONCONTIG_SHAPES = [(17, 33), (5, 7, 9)]

# Backward shapes stay small (the autograd graph is built on the CPU reference
# and the analytic comparison below is elementwise).
_ADD_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]

# aten requires an integral alpha for integral inputs; 2 and -3 exercise both
# signs of the scale factor on the int path.
_ADD_INT_ALPHAS = [2, -3]

# Scalar-scalar add: (a, b, alpha, expected dtype). Both aten and the candidate
# promote the Python scalars to a 0-dim tensor of the natural dtype.
_ADD_SCALAR_SCALAR_CASES = [
    (1.5, -2.5, 0.5, torch.float32),
    (-0.001, 100.001, 2.0, torch.float32),
    (3, 4, 2, torch.int64),
    (-7, 100, -3, torch.int64),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.add callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("add", flag_gems.add)


def _resolve_gems_op_inplace():
    return flag_gems.testing.resolve_gems_op("add_", flag_gems.add_)


def _add_out_adapter(self, other, *, alpha=1, out):
    # Default implementation of the ".out" overload: run the direct add kernel
    # and copy the result into the caller's out buffer. KernelGen's override of
    # "add.out" replaces this adapter with a real out-kernel.
    out.copy_(
        flag_gems.testing.resolve_gems_op("add", flag_gems.add)(
            self, other, alpha=alpha
        )
    )
    return out


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op("add.out", _add_out_adapter)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_FLOAT_DTYPES)
def test_add_tensor_tensor_float_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_INT_DTYPES)
def test_add_tensor_tensor_int_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    # int add is exact and wraps identically on both paths (alpha stays 1).
    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_add_tensor_tensor_bool_value_ranges(shape, value_range):
    inp = tu.make_input(torch.bool, shape, value_range)
    other = tu.make_input(torch.bool, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    # bool add behaves as logical OR; make_input ignores the range for bool.
    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_tensor_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other, alpha=alpha)
    res_out = _resolve_gems_op()(inp, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", _ADD_INT_ALPHAS)
@pytest.mark.parametrize("dtype", _ADD_INT_DTYPES)
def test_add_tensor_tensor_int_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    # aten only accepts an integral alpha for integral inputs; the candidate
    # must reproduce the scaled values exactly.
    ref_out = torch.ops.aten.add(ref_inp, ref_other, alpha=alpha)
    res_out = _resolve_gems_op()(inp, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.add.Scalar(ref_inp, scalar, alpha=alpha)
    res_out = _resolve_gems_op()(inp, scalar, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_scalar_tensor(shape, scalar, alpha, dtype):
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_other = utils.to_reference(other)

    # Scalar-first ordering: aten's tensor-first .Scalar overload computes
    # other + scalar*alpha, which differs from the scalar-first semantics when
    # alpha != 1. Use the scalar-first form torch.ops.aten.add(scalar, tensor)
    # so the reference matches the candidate's scalar + other*alpha.
    ref_out = torch.ops.aten.add(scalar, ref_other, alpha=alpha)
    res_out = _resolve_gems_op()(scalar, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("a,b,alpha,dtype", _ADD_SCALAR_SCALAR_CASES)
def test_add_scalar_scalar(a, b, alpha, dtype):
    # Scalar-scalar add is a pure Python-level promotion: both sides produce a
    # 0-dim tensor of the natural dtype.
    ref_out = torch.ops.aten.add(a, b, alpha=alpha)
    res_out = _resolve_gems_op()(a, b, alpha=alpha)

    assert res_out.dtype == ref_out.dtype == dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("broadcast_pair", _ADD_BROADCAST_PAIRS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + [torch.int32])
def test_add_broadcast(broadcast_pair, dtype):
    shape_a, shape_b = broadcast_pair
    inp = tu.make_input(dtype, shape_a, ["-1", "1"])
    other = tu.make_input(dtype, shape_b, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_nan_inf(dtype):
    # inf + (-inf) -> nan, inf + inf -> inf, 0.0 + -0.0 -> 0.0; 1e30 also
    # covers the overflow-to-inf path in fp16/bf16. equal_nan=True tolerates
    # the nan outputs.
    vals = [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        1.5,
        -2.5,
        1e30,
        -1e30,
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    other = torch.tensor(vals[::-1], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("complex_dtype", _ADD_COMPLEX_DTYPES)
def test_add_complex_value_ranges(shape, value_range, complex_dtype):
    inp = tu.make_input(complex_dtype, shape, value_range)
    other = tu.make_input(complex_dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", [(2, 19, 7)])
@pytest.mark.parametrize("complex_dtype", _ADD_COMPLEX_DTYPES)
@pytest.mark.parametrize("other_type", ["float_tensor", "int_tensor", "int_scalar"])
def test_add_complex_mixed(shape, complex_dtype, other_type):
    # Complex self with real/float tensors and int scalars exercises the
    # candidate's real-view path against mixed promotion rules.
    inp = tu.make_input(complex_dtype, shape, ["-1", "1"])
    if other_type == "float_tensor":
        float_dtype = (
            torch.float32 if complex_dtype == torch.complex64 else torch.float64
        )
        other = tu.make_input(float_dtype, shape, ["-1", "1"])
    elif other_type == "int_tensor":
        other = tu.make_input(torch.int32, shape, ["-1", "1"])
    else:
        other = 3

    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other) if isinstance(other, torch.Tensor) else other

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", [(2, 19, 7)])
@pytest.mark.parametrize("complex_dtype", _ADD_COMPLEX32_DTYPES)
@pytest.mark.parametrize(
    "other_type", ["complex", "float_tensor", "int_tensor", "int_scalar"]
)
def test_add_complex32(shape, complex_dtype, other_type):
    # complex32 (fp16 complex) has no CPU kernel, so the reference is upcast to
    # complex128 before comparing (gems_assert_close casts back internally).
    inp = tu.make_input(complex_dtype, shape, ["-1", "1"])
    if other_type == "complex":
        other = tu.make_input(complex_dtype, shape, ["-1", "1"])
    elif other_type == "float_tensor":
        other = tu.make_input(torch.float16, shape, ["-1", "1"])
    elif other_type == "int_tensor":
        other = tu.make_input(torch.int32, shape, ["-1", "1"])
    else:
        other = 3

    ref_inp = utils.to_reference(inp, True)
    ref_other = (
        utils.to_reference(other, True) if isinstance(other, torch.Tensor) else other
    )

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", _ADD_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    other = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", _ADD_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add_noncontiguous(shape, dtype):
    # transposed views have non-unit strides; the kernel must honor them.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    other = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    res_out = _resolve_gems_op()(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", _ADD_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    other = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    ref_in_grad, ref_other_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), grad_outputs=ref_grad
    )

    # d(a + b)/da == d(a + b)/db == 1, so both gradients are the incoming
    # grad; this validates the reference autograd path itself.
    tu.assert_result_close(ref_in_grad, ref_grad)
    tu.assert_result_close(ref_other_grad, ref_grad)

    # The candidate forward output must match the reference...
    res_out = _resolve_gems_op()(inp, other)
    tu.assert_result_close(res_out, ref_out)

    # ...and, if the candidate kernel advertises autograd support (the current
    # direct kernel does not: res_out.requires_grad is False), its gradients
    # must match the analytic values too.
    if res_out.requires_grad:
        res_in_grad, res_other_grad = torch.autograd.grad(
            res_out, (inp, other), grad_outputs=grad
        )
        tu.assert_result_close(res_in_grad, ref_grad)
        tu.assert_result_close(res_other_grad, ref_grad)


@pytest.mark.add
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward_broadcast(dtype):
    shape_a, shape_b = (2, 3, 5), (5,)
    inp = tu.make_input(dtype, shape_a, ["-1", "1"]).requires_grad_()
    other = tu.make_input(dtype, shape_b, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape_a, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    ref_in_grad, ref_other_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), grad_outputs=ref_grad
    )

    # The gradient w.r.t. the broadcast operand is reduced over the broadcast
    # dims: for (2, 3, 5) vs (5,), g_b == sum(grad, dim=(0, 1)).
    expected_other_grad = ref_grad.sum(dim=(0, 1))
    tu.assert_result_close(ref_in_grad, ref_grad)
    tu.assert_result_close(ref_other_grad, expected_other_grad)

    res_out = _resolve_gems_op()(inp, other)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grad, res_other_grad = torch.autograd.grad(
            res_out, (inp, other), grad_outputs=grad
        )
        tu.assert_result_close(res_in_grad, ref_grad)
        tu.assert_result_close(res_other_grad, expected_other_grad)


@pytest.mark.add_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add__value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add_(ref_inp, ref_other)
    res_out = _resolve_gems_op_inplace()(inp, other)

    # In-place semantics: the call returns the mutated input tensor itself.
    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.add_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add__tensor_scalar(shape, scalar, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.add_(ref_inp, scalar, alpha=alpha)
    res_out = _resolve_gems_op_inplace()(inp, scalar, alpha=alpha)

    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.add_
@pytest.mark.parametrize("broadcast_pair", _ADD_INPLACE_BROADCAST_PAIRS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add__broadcast(broadcast_pair, dtype):
    shape_a, shape_b = broadcast_pair
    inp = tu.make_input(dtype, shape_a, ["-1", "1"])
    other = tu.make_input(dtype, shape_b, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)

    ref_out = torch.ops.aten.add_(ref_inp, ref_other)
    res_out = _resolve_gems_op_inplace()(inp, other)

    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.add_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add_out(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.add.out(ref_inp, ref_other, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, other, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_out_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.add.out(ref_inp, ref_other, alpha=alpha, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, other, alpha=alpha, out=res_out)

    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add_negative
def test_add_rejects_non_broadcastable():
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    other = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.add(inp, other)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, other)


@pytest.mark.add_negative
def test_add_rejects_non_numeric_scalar():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.add(inp, "not-a-number")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, "not-a-number")


@pytest.mark.add_negative
def test_add_requires_two_operands():
    # A single argument hits no overload on either path (the all-scalar form
    # still needs both operands).
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten.add(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
