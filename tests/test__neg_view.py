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

# ``_neg_view`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._neg_view`` and ``-m _neg_view`` both
# work.
setattr(
    pytest.mark,
    "_neg_view",
    MarkDecorator(Mark("_neg_view", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_neg_view(Tensor(a) self) -> Tensor(a) returns an aliasing negative
# view of the input: it shares the input's storage (same shape, strides,
# storage offset and data_ptr) and only toggles the lazy negated bit
# (``is_neg``). No arithmetic happens at view creation, so every storage dtype
# is accepted; materializing the view (e.g. by comparing values) negates the
# elements, so the observed values equal ``-self``. aten only implements that
# negation for the regular numeric dtypes: float8 and bool create the view but
# raise when it is materialized ("neg_cuda not implemented"). For those dtypes,
# check the view metadata and observe storage through an alias with the neg bit
# cleared. Backward compares the sign-flipped upstream gradient with ATen.
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * dtypes: the required spec dtypes, plus the
#     operator's remaining float64/int16/complex64 storage dtypes; the
#     fp8/bool dtypes also have their stored values checked;
#   * shape levels: tu.selected_shapes() (ranks 0-5, selected by --quick) plus
#     a couple of small representative shapes;
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     storage dtype is exercised with negative, positive, extreme and
#     degenerate ranges (tu.make_input clamps unsigned bounds);
#   * edge cases: non-contiguous (strided) inputs, the neg-bit toggle, writing
#     through the returned alias, and nan/inf/+-0.0 special values;
#   * backward: autograd.grad() against ATen for floating and complex inputs,
#     including empty gradients (broadcast does not apply to a unary view op);
#   * negative: a non-tensor input raises on both the aten reference and the
#     candidate.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
# The shape sweep uses a non-degenerate positive range for unsigned storage.
_UNSIGNED_DTYPES = [torch.uint8]
# aten's negation kernel is not implemented for float8/bool. Their negative
# views can be inspected through a second view with the neg bit cleared.
_UNMATERIALIZABLE_DTYPES = _FP8_DTYPES + [torch.bool]
# complex32 is experimental (torch.empty emits a UserWarning) and is not part
# of the required grid; complex64 is the operator's stable complex storage dtype.
_COMPLEX_DTYPES = [torch.complex64]


def _candidate_dtypes():
    # Required spec dtypes first, then the operator's remaining storage dtypes,
    # de-duplicated while preserving order.
    return list(
        dict.fromkeys(
            tu.REQUIRED_DTYPES
            + utils.ALL_FLOAT_DTYPES
            + utils.ALL_INT_DTYPES
            + _COMPLEX_DTYPES
            + utils.BOOL_TYPES
        )
    )


def _basic_range(dtype):
    # Non-degenerate representative range for the shape/dtype sweep.
    return ["0", "max"] if dtype in _UNSIGNED_DTYPES else ["-1", "1"]


_CANDIDATE_DTYPES = _candidate_dtypes()
_VALUE_DTYPES = [
    dtype for dtype in _CANDIDATE_DTYPES if dtype not in _UNMATERIALIZABLE_DTYPES
]
_VIEW_DTYPES = [
    dtype for dtype in _CANDIDATE_DTYPES if dtype in _UNMATERIALIZABLE_DTYPES
]
_ALL_TEST_DTYPES = list(dict.fromkeys(_VALUE_DTYPES + _VIEW_DTYPES))

# Shape levels (0-D up to 5-D), empty layouts and two small shapes.
_NEG_VIEW_SHAPES = list(
    dict.fromkeys([(17,), (12, 13), (0,), (3, 0), (2, 0, 4)] + tu.selected_shapes())
)

# Representative ranks for the full value-range sweep.
_NEG_VIEW_RANGE_SHAPES = tu.selected_shapes()
_NEG_VIEW_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_TOGGLE_SHAPES = [(16, 32), (4, 8, 16)]
_NEG_VIEW_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_NEG_VIEW_BACKWARD_SHAPES = [(16, 64), (7, 13, 29), (0,), (3, 0)]


_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _ALL_TEST_DTYPES
    for value_range in tu.selected_ranges()
]


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "_neg_view", getattr(flag_gems, "_neg_view", None)
    )


def _assert_values_equal(res_out, ref_out):
    # For dtypes without a negation kernel, observe the unchanged storage by
    # flipping the neg bit on both outputs. This only creates another alias;
    # the original outputs and their already-checked view metadata stay intact.
    if ref_out.dtype in _UNMATERIALIZABLE_DTYPES and ref_out.is_neg():
        res_out = torch.ops.aten._neg_view(res_out)
        ref_out = torch.ops.aten._neg_view(ref_out)
    tu.assert_result_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _neg_view returns an aliasing view (Tensor(a)): shape, strides, storage
    # offset, neg state and the shared storage must match aten exactly.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_neg() == ref_out.is_neg()
    # Empty tensors can have equal null pointers without sharing storage.
    assert torch._C._is_alias_of(res_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__neg_view(shape, dtype):
    # Shape levels x every materializable dtype (including the required int8/
    # uint8 dtypes) over a non-degenerate representative value range.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    assert res_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_value_ranges(shape, dtype, value_range):
    # Every storage dtype covers the full value-range grid. Check negative
    # values where materializable, otherwise inspect the underlying storage.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test__neg_view_unmaterializable_dtypes(shape, dtype):
    # float8/bool cannot materialize negative values. Verify both the original
    # view metadata and the unchanged storage through a view with its neg bit off.
    inp = tu.make_input(dtype, shape, ["0", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    # A negative view must preserve the exact strides of a non-contiguous
    # input. Slice on both the test device and the reference device so the two
    # inputs share the same memory layout.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_TOGGLE_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_toggle(shape, dtype):
    # The neg bit is a toggle: applying _neg_view to an already-negated tensor
    # clears the bit and the materialized values come back to the base input.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = tu.to_reference(base)

    inp = torch.ops.aten._neg_view(base)
    ref_inp = torch.ops.aten._neg_view(ref_base)
    assert inp.is_neg()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, base)
    _assert_values_equal(res_out, ref_out)
    assert not res_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(values)

    _assert_view_semantics(res_out, ref_out, values)
    tu.assert_result_equal(res_out, ref_out)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = values == 0
    tu.assert_result_equal(
        torch.signbit(res_out[zeros]), torch.signbit(ref_out[ref_inp == 0])
    )


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__neg_view_mutation(shape, dtype):
    # The result is a true alias of the input (Tensor(a)): writing through the
    # returned view stores the negated value into the shared storage and must
    # be observable on the candidate-side input. The reference runs on an
    # independent clone so the two aliases are validated separately.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_out = _resolve_gems_op()(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    tu.assert_result_equal(res_out, ref_out)
    # fill_ through a neg view writes -2.5 into the base storage, so the input
    # (no neg bit) materializes to -2.5 on both sides.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._neg_view
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_FP8_DTYPES))
)
def test__neg_view_fp8_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    # The view must preserve storage bytes, including signed zeros and NaNs.
    tu.assert_result_equal(inp.view(torch.uint8), ref_inp.view(torch.uint8))


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_BACKWARD_SHAPES)
# FP8 backward calls aten.neg, which has no CPU/CUDA FP8 kernel.
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES + _COMPLEX_DTYPES)
)
def test__neg_view_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _resolve_gems_op()(inp)
    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark._neg_view
def test__neg_view_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
