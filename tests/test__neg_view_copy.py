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

# The KernelGen verification harness stages these files in an isolated copy of
# the FlagGems tree whose parent directory is not on sys.path. Make the
# ``tests``/``benchmark`` packages importable regardless of the harness
# process's sys.path so the relative imports below resolve.
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_neg_view_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._neg_view_copy`` and ``-m
# _neg_view_copy`` both work.
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

# aten::_neg_view_copy(Tensor self) -> Tensor materializes the negative view
# as a fresh contiguous copy: the output contains -self, does not alias the
# input, and never mutates it. Negation flips the sign bit, so every dtype that
# aten::neg supports is exact (the two's-complement wrap at INT_MIN is part of
# the reference contract); bool is rejected by aten::neg with a RuntimeError.
# The .out overload writes into and returns the caller's buffer.
#
# Coverage follows the regular-operator spec adapted to a view_copy op:
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by --quick);
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges;
#   * edge cases: non-contiguous (strided) inputs, empty tensors, and
#     nan/inf/±0.0 special values;
#   * backward: autograd.grad() against the analytic gradient -1 (a unary
#     view_copy op, so broadcast does not apply);
#   * negative: bool inputs and non-tensor inputs raise on the aten reference
#     and must fail on the candidate too.
_NEG_VIEW_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_NEG_VIEW_COPY_RANGE_SHAPES = [(), (256,), (7, 13, 29)]
_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_neg_view_copy" and "_neg_view_copy.out".
    return flag_gems.testing.resolve_gems_op(
        "_neg_view_copy", getattr(flag_gems, "_neg_view_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_neg_view_copy.out", getattr(flag_gems, "_neg_view_copy_out", None)
    )


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
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    # Clone so the post-call equality check below can detect any mutation of
    # the input even when the reference runs on the same device.
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_value_ranges(shape, value_range, dtype):
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
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert ref_ret is ref_out
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out_value_ranges(shape, value_range, dtype):
    # The .out path must reproduce the same sign-flip over every spec range
    # while writing into the caller's buffer (overwriting its previous value).
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

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

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Sign-bit flip: +0.0 negates to -0.0 and -0.0 negates to +0.0.
    assert torch.signbit(res_out[0]).item() and not torch.signbit(res_out[1]).item()


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_non_contiguous(shape, dtype):
    # The copy materializes a fresh contiguous tensor with the same logical
    # shape regardless of the input's strides. Slice on both the test device
    # and the reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
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
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
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
def test__neg_view_copy_rejects_bool():
    # Materializing the negation of a bool tensor is not implemented in aten
    # (RuntimeError), so the candidate must fail too rather than silently
    # treating bool negation as a logical not.
    inp = torch.tensor([True, False], dtype=torch.bool, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark._neg_view_copy
def test__neg_view_copy_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
