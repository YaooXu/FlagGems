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

# The KernelGen verification harness stages these files in an isolated copy
# of the FlagGems tree whose parent directory is not on sys.path. Make the
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
from torch.autograd.forward_ad import dual_level  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

# ``_unpack_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._unpack_dual`` and ``-m _unpack_dual``
# both work.
setattr(
    pytest.mark,
    "_unpack_dual",
    MarkDecorator(Mark("_unpack_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor
# tangent) is the forward-mode AD dual-construction inverse: given a dual tensor
# created at an active forward-mode AD ``level`` (e.g. by
# ``torch.ops.aten._make_dual``), it returns the primal as an aliasing view
# (Tensor(a), same shape, strides, storage offset, dtype and data_ptr) and the
# tangent registered at that level. The level must already be active, so the
# native implementation is only callable inside
# ``torch.autograd.forward_ad.dual_level()``; the caller passes the level that
# the context assigned (its exact value varies across torch versions, so it is
# always taken from the context here). On a plain tensor with no forward tangent
# at ``level`` the tangent component is None while the primal view is still
# returned. No arithmetic happens, so the op is pure metadata/view manipulation:
# primal values, tangent values and the alias contract are all asserted below.
#
# The op performs no data movement, so element counts stay small (<= 96K) and
# ranks 0-5 are covered. Ranks are exercised through a shape parametrization;
# each (shape, dtype) pair is a distinct workload.
UNPACK_DUAL_SHAPES = [
    (),
    (1,),
    (64, 64),
    (20, 320, 15),
    (4, 8, 16, 32),
    (2, 3, 4, 5, 6),
]

# Levels accepted by aten on plain tensors (no forward tangent) are 0, 1 and
# any higher value; 0 is always the documented level for dual tensors created
# inside the outermost ``dual_level()`` context.
UNPACK_DUAL_LEVELS = [0, 1, 3]

# Dual tensors can only be constructed over floating-point or complex primals
# (aten raises on int/bool primals). complex32 is excluded because the torch
# comparison helpers cannot materialize ComplexHalf on the reference device;
# complex64 covers the complex branch exactly.
UNPACK_DUAL_DTYPES = utils.ALL_FLOAT_DTYPES + [torch.complex64]

# Plain tensors have no such restriction, so every storage dtype is valid for
# the tangent-None path.
PLAIN_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES + [torch.complex64]
)


def _make_input(shape, dtype):
    if dtype.is_complex or dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._unpack_dual is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._unpack_dual callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_unpack_dual", getattr(flag_gems, "_unpack_dual", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_primal_view(res_primal, ref_primal, dual):
    # _unpack_dual returns the primal as an aliasing view (Tensor(a)): the
    # observable layout must match aten exactly and the result must share
    # storage with the candidate-side dual tensor.
    assert res_primal.dtype == ref_primal.dtype
    assert res_primal.shape == ref_primal.shape
    assert res_primal.stride() == ref_primal.stride()
    assert res_primal.storage_offset() == ref_primal.storage_offset()
    assert res_primal.data_ptr() == dual.data_ptr()


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", UNPACK_DUAL_SHAPES)
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_dual_tensor(shape, dtype):
    # The main forward-mode AD path: unpack a dual tensor at the level that
    # created it and recover both the primal view and the registered tangent.
    primal = _make_input(shape, dtype)
    tangent = _make_input(shape, dtype)
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        # A dual tensor created with a tangent must yield a tensor tangent, not
        # None.
        assert isinstance(res_tangent_out, torch.Tensor)
        _assert_close(res_primal_out, ref_primal_out, dtype)
        _assert_close(res_tangent_out, ref_tangent_out, dtype)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", UNPACK_DUAL_SHAPES)
@pytest.mark.parametrize("level", UNPACK_DUAL_LEVELS)
@pytest.mark.parametrize("dtype", PLAIN_DTYPES)
def test__unpack_dual_plain_tensor(shape, level, dtype):
    # A plain tensor has no forward tangent at any level: aten still returns the
    # input as an aliasing view, and the tangent component must be None. This
    # exercises the tangent-None contract across every storage dtype.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_inp, level)
    res_primal_out, res_tangent_out = _resolve_gems_op()(inp, level)

    assert ref_tangent_out is None
    assert res_tangent_out is None
    _assert_close(res_primal_out, ref_primal_out, dtype)
    _assert_primal_view(res_primal_out, ref_primal_out, inp)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_non_contiguous(shape, dtype):
    # The aliasing primal view must preserve the exact strides and storage
    # offset of a non-contiguous primal. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout. The
    # tangent may be materialized with a different layout by the forward-AD
    # machinery, so only its values are compared.
    base = _make_input(shape, dtype)
    ref_base = utils.to_reference(base)
    primal = base[..., ::2]
    ref_primal = ref_base[..., ::2]
    tangent = _make_input(primal.shape, dtype)
    ref_tangent = utils.to_reference(tangent)
    assert not primal.is_contiguous()

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        assert res_primal_out.stride() == ref_primal_out.stride()
        assert res_primal_out.storage_offset() == ref_primal_out.storage_offset()
        assert res_primal_out.data_ptr() == primal.data_ptr()
        _assert_close(res_primal_out, ref_primal_out, dtype)
        _assert_close(res_tangent_out, ref_tangent_out, dtype)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_mutation(shape, dtype):
    # The returned primal is a true alias of the dual (Tensor(a)): writing
    # through it must be observable on the candidate-side primal, and the
    # reference must behave identically. The op itself never mutates anything.
    primal = _make_input(shape, dtype)
    tangent = _make_input(shape, dtype)
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, _ = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, _ = _resolve_gems_op()(dual, level)

        ref_primal_out.fill_(2.5)
        res_primal_out.fill_(2.5)

        assert res_primal_out.data_ptr() == dual.data_ptr()
        _assert_close(res_primal_out, ref_primal_out, dtype)
        utils.gems_assert_equal(primal, ref_primal)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_special_values(dtype):
    # A pure alias must preserve every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly through both the
    # primal and the tangent.
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_primal = utils.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(values, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        utils.gems_assert_equal(res_primal_out, ref_primal_out, equal_nan=True)
        utils.gems_assert_equal(res_tangent_out, ref_tangent_out, equal_nan=True)
        assert (
            torch.signbit(res_primal_out[0]).item() == torch.signbit(values[0]).item()
        )
        assert (
            torch.signbit(res_primal_out[1]).item() == torch.signbit(values[1]).item()
        )
