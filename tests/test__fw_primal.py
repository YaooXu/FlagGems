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

# ``_fw_primal`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._fw_primal`` and ``-m _fw_primal`` both
# work.
setattr(
    pytest.mark,
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a) is the forward-mode
# AD view primitive: it returns an aliasing view of ``self`` that shares the
# input's storage (same shape, strides, storage offset, data_ptr and dtype)
# without any arithmetic. Level 0 is the documented, always-valid level; on
# plain tensors (no forward tangent registered) aten also accepts any higher
# level and still returns the input as a view, so the parametrization below
# covers both. Dual tensors created inside ``torch.autograd.forward_ad`` are an
# internal AD-machinery edge case (level > 0 raises there) and are out of scope
# for a generated kernel op.
#
# The op is pure metadata manipulation, so every storage dtype is supported and
# the result compares bit-for-bit. Ranks 0-5 are covered with small element
# counts (<= 96K) since no data movement happens.
FW_PRIMAL_SHAPES = [
    (),
    (1,),
    (64, 64),
    (20, 320, 15),
    (4, 8, 16, 32),
    (2, 3, 4, 5, 6),
]

FW_PRIMAL_LEVELS = [0, 1, 3]

FW_PRIMAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
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
    # until flag_gems._fw_primal is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._fw_primal callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_fw_primal", getattr(flag_gems, "_fw_primal", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _fw_primal returns an aliasing view (Tensor(a)): the observable layout
    # must match aten exactly and the result must share storage with the input.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", FW_PRIMAL_SHAPES)
@pytest.mark.parametrize("level", FW_PRIMAL_LEVELS)
@pytest.mark.parametrize("dtype", FW_PRIMAL_DTYPES)
def test__fw_primal(shape, level, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", FW_PRIMAL_DTYPES)
def test__fw_primal_non_contiguous(shape, level, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = _make_input(shape, dtype)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test__fw_primal_mutation(shape, dtype):
    # The result is a true alias of the input: writing through the returned
    # view must be observable on the candidate-side input tensor, and the
    # reference must behave identically.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    level = 0

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    if dtype == torch.bool:
        res_out.fill_(True)
        ref_out.fill_(True)
    elif dtype.is_floating_point:
        res_out.fill_(2.5)
        ref_out.fill_(2.5)
    else:
        res_out.fill_(7)
        ref_out.fill_(7)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
