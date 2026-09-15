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

# ``_neg_view_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._neg_view_copy`` and
# ``-m _neg_view_copy`` both work.
setattr(
    pytest.mark,
    "_neg_view_copy",
    MarkDecorator(Mark("_neg_view_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_neg_view_copy_out",
    MarkDecorator(Mark("_neg_view_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_neg_view_copy(Tensor self) -> Tensor materializes the negative view as
# a fresh contiguous copy: the result holds ``-self``, does NOT alias the input
# and never mutates it. Negation flips the sign bit, so every dtype that
# aten::neg supports is exact (for integers the two's-complement wrap at INT_MIN
# is part of the reference contract). aten's negation kernel is only implemented
# for the regular numeric dtypes: float8 (``"neg_cuda" not implemented``) and
# bool are rejected and covered by the negative cases.
#
# The .out overload writes into (and returns) the caller's buffer; it is a real,
# callable ATen overload on this backend, so it is exercised directly.
#
# Coverage follows the regular-operator spec adapted to a view_copy op:
#   * dtypes: the spec-required dtypes (int8, uint8,
#     float32, bfloat16, float16, int32, int64) plus the operator's remaining
#     numeric storage dtypes (int16, float64);
#   * shape levels: tu.selected_shapes() (0~5 dims, selected by --quick) plus a
#     couple of small representative shapes;
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (tu.make_input clamps unsigned bounds);
#   * edge cases: non-contiguous (strided) inputs, empty tensors and
#     nan/inf/+-0.0 special values;
#   * backward: autograd.grad() against the ATen gradient (a unary
#     view_copy op, so broadcast does not apply);
#   * negative: unsupported dtypes (float8/bool) and a non-tensor input must
#     fail on the candidate exactly like the aten reference.
_UNSIGNED_DTYPES = [torch.uint8]


def _candidate_dtypes():
    # Required spec dtypes first, then the operator's remaining numeric storage
    # dtypes, de-duplicated while preserving order.
    return list(
        dict.fromkeys(
            tu.REQUIRED_DTYPES
            + utils.ALL_FLOAT_DTYPES
            + utils.ALL_INT_DTYPES
            + utils.BOOL_TYPES
        )
    )


def _basic_range(dtype):
    # Non-degenerate representative range for the shape/dtype sweep.
    return ["0", "max"] if dtype in _UNSIGNED_DTYPES else ["-1", "1"]


_CANDIDATE_DTYPES = _candidate_dtypes()
_NEG_VIEW_COPY_DTYPES = [
    dtype
    for dtype in _CANDIDATE_DTYPES
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bool)
]
_UNSUPPORTED_DTYPES = [
    dtype for dtype in _CANDIDATE_DTYPES if dtype not in _NEG_VIEW_COPY_DTYPES
]

# Shape levels (0-D up to 5-D) plus two small representative shapes.
_NEG_VIEW_COPY_SHAPES = list(dict.fromkeys([(17,), (12, 13)] + tu.selected_shapes()))
# Representative ranks for the full value-range sweep.
_NEG_VIEW_COPY_RANGE_SHAPES = tu.selected_shapes()
_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEG_VIEW_COPY_DTYPES
    for value_range in tu.selected_ranges()
]


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "_neg_view_copy", getattr(flag_gems, "_neg_view_copy", None)
    )


def _make_out(shape, dtype, device):
    # Garbage-prefilled buffer: the copy must overwrite every element.
    return torch.full(shape, 7, dtype=dtype, device=device)


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    # _neg_view_copy returns a fresh contiguous copy: same shape/dtype, no
    # aliasing of the input, no neg bit, and the input is never mutated.
    assert res_out.is_contiguous()
    assert not res_out.is_neg()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    # Shape levels x every supported dtype over a non-degenerate representative
    # value range.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_value_ranges(shape, dtype, value_range):
    # The op only flips the sign bit, so the full spec range sweep (negative,
    # positive, extreme and degenerate ranges) must round-trip exactly. For
    # integers the reference wraps at INT_MIN (two's complement); the candidate
    # is held to the same behavior by comparing against the reference.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = tu.to_reference(inp)

    ref_out = _make_out(shape, ref_inp.dtype, ref_inp.device)
    out = _make_out(shape, dtype, flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op()(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_out_value_ranges(shape, dtype, value_range):
    # The .out path must reproduce the same sign-flip over every spec range
    # while overwriting the caller's buffer.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = _make_out(shape, ref_inp.dtype, ref_inp.device)
    out = _make_out(shape, dtype, flag_gems.device)

    torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op()(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_copy_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(values)

    tu.assert_result_equal(res_out, ref_out)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = values == 0
    tu.assert_result_equal(
        torch.signbit(res_out[zeros]), torch.signbit(ref_out[ref_inp == 0])
    )


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_non_contiguous(shape, dtype):
    # The copy materializes a fresh contiguous tensor with the same logical
    # shape regardless of the input's strides. Transpose on both the test device
    # and the reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_empty(shape, dtype):
    # Zero-element tensors must be handled without out-of-bounds accesses.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__neg_view_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _resolve_gems_op()(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__neg_view_copy_rejects_unsupported_dtypes(dtype):
    # The underlying negation is not implemented for these dtypes (float8 raise
    # "neg_cuda not implemented", bool likewise), so the candidate must reject
    # them too rather than silently producing a wrong result.
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark._neg_view_copy
def test__neg_view_copy_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float fails to match any schema);
    # the candidate must fail too rather than silently accept scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
