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
from torch.autograd.forward_ad import dual_level

import flag_gems

from . import accuracy_utils as utils

# ``_make_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._make_dual`` and ``-m _make_dual``
# both work.
setattr(
    pytest.mark,
    "_make_dual",
    MarkDecorator(Mark("_make_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
# is the forward-mode AD dual-construction primitive: it attaches ``tangent``
# to an aliasing view of ``primal`` at the forward-mode AD level ``level``. The
# level must already be active, so the native implementation is only callable
# inside ``torch.autograd.forward_ad.dual_level()``; the caller passes the
# level that the context assigned (its exact value varies across torch
# versions, so it is always taken from the context here). The observable value
# of the result is exactly ``primal`` (no arithmetic happens), so the candidate
# must reproduce primal's shape, dtype and storage layout, must alias primal
# (Tensor(a)), and must preserve the tangent: unpacking the result with
# ``torch.autograd.forward_ad.unpack_dual`` must recover primal and the
# original tangent unchanged. The primal must be floating-point or complex
# (aten raises on int/bool primals) and the tangent must match the primal's
# size; both constraints are respected below.
#
# The op performs no data movement, so element counts stay small (<= 96K) and
# ranks 0-5 are covered. Ranks are exercised through a shape parametrization;
# each (shape, dtype) pair is a distinct workload.
MAKE_DUAL_SHAPES = [
    (),
    (1,),
    (64, 64),
    (20, 320, 15),
    (4, 8, 16, 32),
    (2, 3, 4, 5, 6),
]

# Primal must be floating-point or complex. complex32 is excluded because the
# torch comparison helpers cannot materialize ComplexHalf on the reference
# device; complex64 covers the complex branch exactly.
MAKE_DUAL_DTYPES = utils.ALL_FLOAT_DTYPES + [torch.complex64]


def _make_input(shape, dtype):
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._make_dual is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._make_dual callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_make_dual", getattr(flag_gems, "_make_dual", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _make_dual returns an aliasing view (Tensor(a)) of the primal: the
    # observable layout must match aten exactly and the result must share
    # storage with the candidate-side primal.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()


def _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype):
    # The whole purpose of the op is to produce a dual tensor: unpacking the
    # result must recover the primal value and the unchanged input tangent.
    # (unpack_dual of a plain non-dual tensor returns a None tangent, so the
    # candidate cannot skip the dual wrapping.)
    res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
    ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
    _assert_close(res_primal, ref_primal, dtype)
    _assert_close(res_tangent, ref_tangent_out, dtype)
    _assert_close(res_tangent, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", MAKE_DUAL_SHAPES)
@pytest.mark.parametrize("dtype", MAKE_DUAL_DTYPES)
def test__make_dual(shape, dtype):
    inp = _make_input(shape, dtype)
    tangent = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_close(res_out, ref_out, dtype)
        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", MAKE_DUAL_DTYPES)
def test__make_dual_non_contiguous(shape, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous primal. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = _make_input(shape, dtype)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    tangent = _make_input(inp.shape, dtype)
    ref_tangent = utils.to_reference(tangent)
    assert not inp.is_contiguous()

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_close(res_out, ref_out, dtype)
        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_mutation(shape, dtype):
    # The result is a true alias of the primal (Tensor(a)): writing through the
    # returned view must be observable on the candidate-side input, and the
    # reference must behave identically. The op itself never mutates the primal.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    tangent = _make_input(shape, dtype)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        ref_out.fill_(2.5)
        res_out.fill_(2.5)

        _assert_close(res_out, ref_out, dtype)
        assert res_out.data_ptr() == inp.data_ptr()
        utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_special_values(dtype):
    # A pure alias must preserve every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(values, tangent, level)

        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
        assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
        assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()
