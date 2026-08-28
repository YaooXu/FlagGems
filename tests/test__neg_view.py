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
# (``is_neg``). Materializing the view (e.g. by comparing values) negates the
# elements, so the observed values equal ``-self``. No arithmetic happens at
# view creation, so every storage dtype is supported; bool is excluded only
# because materializing the negation of a bool tensor raises in aten.
NEG_VIEW_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._neg_view is registered; resolution order is: (1) override,
    # (2) the direct flag_gems._neg_view callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_neg_view", getattr(flag_gems, "_neg_view", None)
    )


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


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
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    assert ref_out.is_neg()


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    # A negative view must preserve the exact strides of a non-contiguous
    # input. Slice on both the test device and the reference device so the two
    # inputs share the same memory layout.
    base = _make_input(shape, dtype)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._neg_view
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", NEG_VIEW_DTYPES)
def test__neg_view_toggle(shape, dtype):
    # The neg bit is a toggle: applying _neg_view to an already-negated tensor
    # clears the bit and the materialized values come back to the base input.
    base = _make_input(shape, dtype)
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
