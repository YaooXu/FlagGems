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


import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

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
# is supported; materializing the view (e.g. by comparing values) negates the
# elements, so the observed values equal ``-self``. bool is excluded because
# materializing the negation of a bool tensor raises in aten ("neg_cpu not
# implemented for Bool"). The view is autograd-aware: materializing computes
# ``-self``, so ``d(-x)/dx == -1``, which the backward test validates against
# the analytic value.
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by --quick);
#   * value ranges: tu.selected_ranges() over representative ranks, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the aliasing view round-trips all of them exactly
#     through the negated materialization);
#   * edge cases: non-contiguous (strided) inputs, the neg-bit toggle, writing
#     through the returned alias, and nan/inf/±0.0 special values;
#   * backward: autograd.grad() through the neg view against the analytic
#     gradient -1 (broadcast does not apply to a unary view op);
#   * negative: a non-tensor input raises on both the aten reference and the
#     candidate.
NEG_VIEW_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_NEG_VIEW_RANGE_SHAPES = [(), (256,), (7, 13, 29)]

_NEG_VIEW_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_NEG_VIEW_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_NEG_VIEW_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._neg_view is registered; resolution order is: (1) override,
    # (2) the direct flag_gems._neg_view callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_neg_view", getattr(flag_gems, "_neg_view", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _neg_view returns an aliasing view (Tensor(a)): shape, strides and
    # storage offset are preserved, the neg bit matches the reference, and the
    # result shares storage with the candidate-side input.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    assert ref_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip exactly through the negated materialization.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    # A negative view must preserve the exact strides of a non-contiguous
    # input. Slice on both the test device and the reference device so the two
    # inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view_toggle(shape, dtype):
    # The neg bit is a toggle: applying _neg_view to an already-negated tensor
    # clears the bit and the materialized values come back to the base input.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)

    inp = torch.ops.aten._neg_view(base)
    ref_inp = torch.ops.aten._neg_view(ref_base)
    assert inp.is_neg()
    assert ref_inp.is_neg()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
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

    _assert_close(res_out, ref_out, dtype)
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
    _assert_close(res_out, ref_out, dtype)
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
