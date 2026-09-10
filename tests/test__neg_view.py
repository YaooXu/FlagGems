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

import math

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
# raise when it is materialized ("neg_cuda not implemented"). Those two dtypes
# are therefore exercised through the view contract alone - exactly the part
# the operator actually promises. The view is autograd-aware: materializing
# computes ``-self``, so ``d(-x)/dx == -1``, which the backward test validates
# against the analytic value.
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * dtypes: the required spec dtypes probed on the active device, plus the
#     operator's remaining float64/int16/complex64 storage dtypes; the
#     unmaterializable fp8/bool dtypes get a dedicated view-semantics sweep;
#   * shape levels: tu.selected_shapes() (ranks 0-5, selected by --quick) plus
#     a couple of small representative shapes;
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     materializable dtype is exercised with negative, positive, extreme and
#     degenerate ranges (unsigned dtypes only accept ranges whose lower bound
#     is non-negative, because they cannot represent negatives);
#   * edge cases: non-contiguous (strided) inputs, the neg-bit toggle, writing
#     through the returned alias, and nan/inf/+-0.0 special values;
#   * backward: autograd.grad() through the neg view against the analytic
#     gradient -1 (broadcast does not apply to a unary view op);
#   * negative: a non-tensor input raises on both the aten reference and the
#     candidate.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
# Unsigned dtypes cannot represent negative values; tu.make_input only accepts
# ranges whose lower bound resolves to >= 0 for them.
_UNSIGNED_DTYPES = [torch.uint8]
# aten's negation kernel is not implemented for float8/bool: the view is
# creatable and inspectable but its values can never be read.
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


def _view_works(dtype):
    # Probe the real aten op on the active device; any exception means the
    # negative view is not available for this storage dtype on this backend.
    try:
        probe = tu.make_input(dtype, (2, 2), _basic_range(dtype))
        return bool(torch.ops.aten._neg_view(probe).is_neg())
    except Exception:
        return False


def _materializes(dtype):
    # The values of the view are only observable if aten can negate this dtype.
    try:
        probe = tu.make_input(dtype, (2, 2), _basic_range(dtype))
        view = torch.ops.aten._neg_view(probe)
        _ = view + 0
        return True
    except Exception:
        return False


_CANDIDATE_DTYPES = [dtype for dtype in _candidate_dtypes() if _view_works(dtype)]
_VALUE_DTYPES = [dtype for dtype in _CANDIDATE_DTYPES if _materializes(dtype)]
_VIEW_DTYPES = [
    dtype for dtype in _CANDIDATE_DTYPES if dtype in _UNMATERIALIZABLE_DTYPES
]
if not _VALUE_DTYPES:
    _VALUE_DTYPES = [torch.float32]
if not _VIEW_DTYPES:
    _VIEW_DTYPES = [torch.bool]
_ALL_TEST_DTYPES = list(dict.fromkeys(_VALUE_DTYPES + _VIEW_DTYPES))

# Shape levels (0-D up to 5-D) plus two small representative shapes.
_NEG_VIEW_SHAPES = list(dict.fromkeys([(17,), (12, 13)] + tu.selected_shapes()))

# Representative ranks for the full value-range sweep.
_NEG_VIEW_RANGE_SHAPES = [(), (256,), (7, 13, 29)]
_NEG_VIEW_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_TOGGLE_SHAPES = [(16, 32), (4, 8, 16)]
_NEG_VIEW_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_NEG_VIEW_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _ranges_for(dtype):
    # The five spec ranges, minus the ones an unsigned dtype cannot represent.
    ranges = tu.selected_ranges()
    if dtype in _UNSIGNED_DTYPES:
        return [rng for rng in ranges if rng[0] == "0"]
    return ranges


_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _VALUE_DTYPES
    for value_range in _ranges_for(dtype)
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems._neg_view callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_neg_view", getattr(flag_gems, "_neg_view", None)
    )


def _assert_values_close(res_out, ref_out, dtype):
    # fp8/bool can never be materialized by aten, so their values are not read.
    if dtype in _VIEW_DTYPES:
        return
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _neg_view returns an aliasing view (Tensor(a)): shape, strides, storage
    # offset, neg state and the shared storage must match aten exactly.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__neg_view(shape, dtype):
    # Shape levels x every materializable dtype (including the required int8/
    # uint8 dtypes) over a non-degenerate representative value range.
    inp = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_values_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    assert ref_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_RANGE_SHAPES)
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_value_ranges(shape, dtype, value_range):
    # The op never reads or transforms the stored values, so the full spec range
    # sweep must round-trip exactly through the negated materialization.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_values_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VIEW_DTYPES)
def test__neg_view_unmaterializable_dtypes(shape, dtype):
    # float8/bool: aten creates the negative view but raises when it is
    # materialized, so only the view contract (shape/stride/offset/data_ptr and
    # the neg bit) can be verified - which is precisely what the op promises.
    inp = tu.make_input(dtype, shape, ["0", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    assert ref_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    # A negative view must preserve the exact strides of a non-contiguous
    # input. Slice on both the test device and the reference device so the two
    # inputs share the same memory layout.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_values_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_TOGGLE_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_toggle(shape, dtype):
    # The neg bit is a toggle: applying _neg_view to an already-negated tensor
    # clears the bit and the materialized values come back to the base input.
    base = tu.make_input(dtype, shape, _basic_range(dtype))
    ref_base = utils.to_reference(base)

    inp = torch.ops.aten._neg_view(base)
    ref_inp = torch.ops.aten._neg_view(ref_base)
    assert inp.is_neg()
    assert ref_inp.is_neg()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_values_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, base)
    assert not res_out.is_neg()
    assert not ref_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__neg_view_special_values(dtype):
    # Materializing the view flips every sign: +inf <-> -inf, nan stays nan,
    # +0.0 <-> -0.0. equal_nan=True tolerates the nan output; copysign pins the
    # sign of the two zero outputs (the sign bit is indistinguishable in a
    # plain value comparison).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(values)

    _assert_view_semantics(res_out, ref_out, values)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    items = res_out.cpu().tolist()
    assert math.isinf(items[0]) and items[0] < 0  # +inf -> -inf
    assert math.isinf(items[1]) and items[1] > 0  # -inf -> +inf
    assert math.isnan(items[2])  # nan -> nan
    assert math.copysign(1.0, items[3]) == -1.0  # +0.0 -> -0.0
    assert math.copysign(1.0, items[4]) == 1.0  # -0.0 -> +0.0


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__neg_view_mutation(shape, dtype):
    # The result is a true alias of the input (Tensor(a)): writing through the
    # returned view stores the negated value into the shared storage and must
    # be observable on the candidate-side input. The reference runs on an
    # independent clone so the two aliases are validated separately.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    res_out = _resolve_gems_op()(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()
    # fill_ through a neg view writes -2.5 into the base storage, so the input
    # (no neg bit) materializes to -2.5 on both sides.
    tu.assert_result_close(inp, ref_inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__neg_view_backward(shape, dtype):
    # Materializing the view computes -x, so d(-x)/dx == -1: the reference
    # gradient must match the analytic value. The candidate is validated on the
    # same contract when it advertises autograd support (a true view of a leaf
    # carries requires_grad through the view machinery; a materializing kernel
    # would not).
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]
    expected_in_grad = -ref_grad
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward output must match the reference...
    res_out = _resolve_gems_op()(inp)
    _assert_values_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)

    # ...and, if the candidate advertises autograd support, its gradient must
    # match the analytic value too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark._neg_view
def test__neg_view_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
