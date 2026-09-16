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

_ADD_INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]

_ADD_DTYPES = utils.ALL_FLOAT_DTYPES + _ADD_INT_DTYPES + utils.BOOL_TYPES

_ADD_COMPLEX_DTYPES = [torch.complex64, torch.complex128]

# Broadcast pairs in both operand orders.
_ADD_BROADCAST_PAIRS = [
    ((20, 320, 15), (15,)),
    ((20, 320, 15), (20, 1, 15)),
    ((20, 320, 15), (1, 320, 1)),
    ((15,), (20, 320, 15)),
    ((20, 1, 15), (20, 320, 15)),
    ((1, 320, 1), (20, 320, 15)),
]

# In-place addition requires self to have the broadcast result shape.
_ADD_INPLACE_BROADCAST_PAIRS = _ADD_BROADCAST_PAIRS[:3]

# Integral inputs require an integral alpha.
_ADD_INT_ALPHAS = tu.selected_cases([0, 1, 2, -3], quick=[1])

# (a, b, alpha, expected dtype).
_ADD_SCALAR_SCALAR_CASES = [
    (1.5, -2.5, 0.5, torch.float32),
    (-0.001, 100.001, 2.0, torch.float32),
    (3, 4, 2, torch.int64),
    (-7, 100, -3, torch.int64),
]


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_add_tensor_tensor_float_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_INT_DTYPES)
def test_add_tensor_tensor_int_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    # int add is exact and wraps identically on both paths (alpha stays 1).
    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_add_tensor_tensor_bool_value_ranges(shape, value_range):
    inp = tu.make_input(torch.bool, shape, value_range)
    other = tu.make_input(torch.bool, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    # bool add behaves as logical OR; make_input ignores the range for bool.
    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", tu.selected_cases([0, 1, *utils.SCALARS], quick=[1]))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_tensor_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", _ADD_INT_ALPHAS)
@pytest.mark.parametrize("dtype", _ADD_INT_DTYPES)
def test_add_tensor_tensor_int_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    # aten only accepts an integral alpha for integral inputs; the candidate
    # must reproduce the scaled values exactly.
    ref_out = torch.ops.aten.add(ref_inp, ref_other, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", [0, 1, *utils.SCALARS])
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES))
def test_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.add.Scalar(ref_inp, scalar, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, scalar, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", [0, 1, *utils.SCALARS])
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES))
def test_add_scalar_tensor(shape, scalar, alpha, dtype):
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_other = tu.to_reference(other)

    # Scalar-first ordering: aten's tensor-first .Scalar overload computes
    # other + scalar*alpha, which differs from the scalar-first semantics when
    # alpha != 1. Use the scalar-first form torch.ops.aten.add(scalar, tensor)
    # so the reference matches the candidate's scalar + other*alpha.
    ref_out = torch.ops.aten.add(scalar, ref_other, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(scalar, other, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("a,b,alpha,dtype", tu.selected_cases(_ADD_SCALAR_SCALAR_CASES))
def test_add_scalar_scalar(a, b, alpha, dtype):
    ref_out = torch.ops.aten.add(a, b, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(a, b, alpha=alpha)

    assert res_out.dtype == ref_out.dtype == dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("broadcast_pair", _ADD_BROADCAST_PAIRS)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES + [torch.int32]))
def test_add_broadcast(broadcast_pair, dtype):
    shape_a, shape_b = broadcast_pair
    inp = tu.make_input(dtype, shape_a, ["-1", "1"])
    other = tu.make_input(dtype, shape_b, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_ADD_DTYPES))
)
@pytest.mark.parametrize("shift", [0, 1])
def test_add_nan_inf(dtype, scenario, shift):
    inp = tu.make_special_input(dtype, scenario)
    # Aligned values add same-sign infinities; shifting pairs +inf with -inf.
    other = inp.roll(shift)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("complex_dtype", _ADD_COMPLEX_DTYPES)
def test_add_complex_value_ranges(shape, value_range, complex_dtype):
    inp = tu.make_input(complex_dtype, shape, value_range)
    other = tu.make_input(complex_dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", [(2, 19, 7)])
@pytest.mark.parametrize("complex_dtype", _ADD_COMPLEX_DTYPES)
@pytest.mark.parametrize("other_type", ["float_tensor", "int_tensor", "int_scalar"])
def test_add_complex_mixed(shape, complex_dtype, other_type):
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

    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other) if isinstance(other, torch.Tensor) else other

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

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
@pytest.mark.parametrize("complex_dtype", [torch.complex32])
@pytest.mark.parametrize(
    "other_type",
    (
        tu.selected_cases(
            ["complex", "float_tensor", "int_tensor", "int_scalar"], quick=["complex"]
        )
    ),
)
def test_add_complex32(shape, complex_dtype, other_type):
    # Upcast the reference to complex128; gems_assert_close casts back for comparison.
    inp = tu.make_input(complex_dtype, shape, ["-1", "1"])
    if other_type == "complex":
        other = tu.make_input(complex_dtype, shape, ["-1", "1"])
    elif other_type == "float_tensor":
        other = tu.make_input(torch.float16, shape, ["-1", "1"])
    elif other_type == "int_tensor":
        other = tu.make_input(torch.int32, shape, ["-1", "1"])
    else:
        other = 3

    ref_inp = tu.to_reference(inp, True)
    ref_other = (
        tu.to_reference(other, True) if isinstance(other, torch.Tensor) else other
    )

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", [(0,), (4, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    other = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", [(17, 33), (5, 7, 9)])
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add_noncontiguous(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    other = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", [(16, 64), (7, 13, 29)])
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_add_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    other = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    ref_in_grad, ref_other_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), grad_outputs=ref_grad
    )

    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)
    tu.assert_result_close(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad, res_other_grad = torch.autograd.grad(
        res_out, (inp, other), grad_outputs=grad
    )
    tu.assert_result_close(res_in_grad, ref_in_grad)
    tu.assert_result_close(res_other_grad, ref_other_grad)


@pytest.mark.add
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_add_backward_broadcast(dtype):
    shape_a, shape_b = (2, 3, 5), (5,)
    inp = tu.make_input(dtype, shape_a, ["-1", "1"]).requires_grad_()
    other = tu.make_input(dtype, shape_b, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape_a, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.add(ref_inp, ref_other)
    ref_in_grad, ref_other_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), grad_outputs=ref_grad
    )

    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_out = gems_op(inp, other)
    tu.assert_result_close(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad, res_other_grad = torch.autograd.grad(
        res_out, (inp, other), grad_outputs=grad
    )
    tu.assert_result_close(res_in_grad, ref_in_grad)
    tu.assert_result_close(res_other_grad, ref_other_grad)


@pytest.mark.add_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_add__value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    other = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add_(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add_")
    res_out = gems_op(inp, other)

    # In-place semantics: the call returns the mutated input tensor itself.
    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.add_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", tu.selected_cases([0, 1, *utils.SCALARS], quick=[1]))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add__tensor_scalar(shape, scalar, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.add_(ref_inp, scalar, alpha=alpha)
    gems_op = flag_gems.testing.resolve_gems_op("add_")
    res_out = gems_op(inp, scalar, alpha=alpha)

    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.add_
@pytest.mark.parametrize("broadcast_pair", _ADD_INPLACE_BROADCAST_PAIRS)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES))
def test_add__broadcast(broadcast_pair, dtype):
    shape_a, shape_b = broadcast_pair
    inp = tu.make_input(dtype, shape_a, ["-1", "1"])
    other = tu.make_input(dtype, shape_b, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.ops.aten.add_(ref_inp, ref_other)
    gems_op = flag_gems.testing.resolve_gems_op("add_")
    res_out = gems_op(inp, other)

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
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten.add.out(ref_inp, ref_other, out=ref_out)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_ret = gems_op(inp, other, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("alpha", tu.selected_cases([0, 1, *utils.SCALARS], quick=[1]))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_out_alpha(shape, alpha, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    other = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_other = tu.to_reference(other)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    torch.ops.aten.add.out(ref_inp, ref_other, alpha=alpha, out=ref_out)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    res_ret = gems_op(inp, other, alpha=alpha, out=res_out)

    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.add_negative
def test_add_rejects_non_broadcastable():
    inp = tu.make_input(torch.float32, (2, 3), ["-1", "1"])
    other = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.add(inp, other)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(inp, other)


@pytest.mark.add_negative
def test_add_rejects_non_numeric_scalar():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.add(inp, "not-a-number")
    gems_op = flag_gems.testing.resolve_gems_op("add")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(inp, "not-a-number")


@pytest.mark.add_negative
def test_add_requires_two_operands():
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten.add(3.14)
    gems_op = flag_gems.testing.resolve_gems_op("add")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(3.14)
