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
# bool are rejected, so the supported-dtype set is probed at import time instead
# of guessed, and those rejected dtypes become the negative cases.
#
# The .out overload writes into (and returns) the caller's buffer; it is a real,
# callable ATen overload on this backend, so it is exercised directly.
#
# Coverage follows the regular-operator spec adapted to a view_copy op:
#   * dtypes: the spec-required dtypes probed on the active device (int8, uint8,
#     float32, bfloat16, float16, int32, int64) plus the operator's remaining
#     numeric storage dtypes (int16, float64);
#   * shape levels: tu.selected_shapes() (0~5 dims, selected by --quick) plus a
#     couple of small representative shapes;
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (unsigned dtypes only accept ranges whose lower bound is
#     non-negative, because they cannot represent negatives);
#   * edge cases: non-contiguous (strided) inputs, empty tensors and
#     nan/inf/+-0.0 special values;
#   * backward: autograd.grad() against the analytic gradient -1 (a unary
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


def _op_works(dtype):
    # Probe the real aten op on the active device; any exception means the copy
    # (i.e. the underlying negation) is not available for this dtype.
    try:
        probe = tu.make_input(dtype, (2, 2), _basic_range(dtype))
        torch.ops.aten._neg_view_copy(probe)
        return True
    except Exception:
        return False


_CANDIDATE_DTYPES = _candidate_dtypes()
_NEG_VIEW_COPY_DTYPES = [dtype for dtype in _CANDIDATE_DTYPES if _op_works(dtype)]
if not _NEG_VIEW_COPY_DTYPES:
    _NEG_VIEW_COPY_DTYPES = [torch.float32]
_UNSUPPORTED_DTYPES = [
    dtype for dtype in _CANDIDATE_DTYPES if dtype not in _NEG_VIEW_COPY_DTYPES
]

# Shape levels (0-D up to 5-D) plus two small representative shapes.
_NEG_VIEW_COPY_SHAPES = list(dict.fromkeys([(17,), (12, 13)] + tu.selected_shapes()))
# Representative ranks for the full value-range sweep.
_NEG_VIEW_COPY_RANGE_SHAPES = [(), (256,), (7, 13, 29)]
_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _ranges_for(dtype):
    # The five spec ranges, minus the ones an unsigned dtype cannot represent.
    ranges = tu.selected_ranges()
    if dtype in _UNSIGNED_DTYPES:
        return [rng for rng in ranges if rng[0] == "0"]
    return ranges


_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEG_VIEW_COPY_DTYPES
    for value_range in _ranges_for(dtype)
]


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default overload is reached through the public operator name
    # "_neg_view_copy".
    return flag_gems.testing.resolve_gems_op(
        "_neg_view_copy", getattr(flag_gems, "_neg_view_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_neg_view_copy.out", getattr(flag_gems, "_neg_view_copy_out", None)
    )


def _make_out(shape, dtype, device):
    # Garbage-prefilled buffer: the copy must overwrite every element.
    return torch.full(shape, 7, dtype=dtype, device=device)


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype):
    # _neg_view_copy returns a fresh contiguous copy: same shape/dtype, no
    # aliasing of the input, no neg bit, and the input is never mutated.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert not res_out.is_neg()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    # Shape levels x every supported dtype over a non-degenerate representative
    # value range.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    # Clone so the post-call equality check below can detect any mutation of
    # the input even when the reference runs on the same device.
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_value_ranges(shape, dtype, value_range):
    # The op only flips the sign bit, so the full spec range sweep (negative,
    # positive, extreme and degenerate ranges) must round-trip exactly. For
    # integers the reference wraps at INT_MIN (two's complement); the candidate
    # is held to the same behavior by comparing against the reference.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _make_out(shape, ref_inp.dtype, ref_inp.device)
    out = _make_out(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert ref_ret is ref_out
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_out_value_ranges(shape, dtype, value_range):
    # The .out path must reproduce the same sign-flip over every spec range
    # while overwriting the caller's buffer.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _make_out(shape, ref_inp.dtype, ref_inp.device)
    out = _make_out(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert ref_ret is ref_out
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
    tu.assert_result_close(out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__neg_view_copy_special_values(dtype):
    # Negation flips the sign bit, so signed zero, infinities and NaN must be
    # preserved exactly (including the -0.0 sign).
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(values)

    tu.assert_result_close(res_out, ref_out)
    # Sign-bit flip: +0.0 negates to -0.0 and -0.0 negates to +0.0.
    assert torch.signbit(res_out[0]).item() and not torch.signbit(res_out[1]).item()


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_non_contiguous(shape, dtype):
    # The copy materializes a fresh contiguous tensor with the same logical
    # shape regardless of the input's strides. Transpose on both the test device
    # and the reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = utils.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_empty(shape, dtype):
    # Zero-element tensors must be handled without out-of-bounds accesses.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__neg_view_copy_backward(shape, dtype):
    # Materializing the negative view computes -x, so d(-x)/dx == -1: the
    # reference gradient must match the analytic value. The candidate is
    # validated on the same contract when it advertises autograd support.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]
    expected_in_grad = -ref_grad
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward output must match the reference...
    res_out = _resolve_gems_op()(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)

    # ...and, if the candidate advertises autograd support, its gradient must
    # match the analytic value too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize(
    "dtype", _UNSUPPORTED_DTYPES if _UNSUPPORTED_DTYPES else [torch.bool]
)
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
