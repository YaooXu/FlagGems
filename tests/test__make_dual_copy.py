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

# ``_make_dual_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._make_dual_copy`` and ``-m
# _make_dual_copy`` both work.
setattr(
    pytest.mark,
    "_make_dual_copy",
    MarkDecorator(Mark("_make_dual_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_make_dual_copy_out",
    MarkDecorator(Mark("_make_dual_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_make_dual_copy(Tensor primal, Tensor tangent, int level) -> Tensor is
# the view_copy materialization of _make_dual: it returns a fresh contiguous
# copy of the primal (Tensor(a) -> Tensor), not an aliasing view. For the
# plain tensors used here the native CompositeExplicitAutograd implementation
# is only reachable through the dispatcher for inference tensors (it is
# guarded by an internal InferenceMode assert), so the reference must run
# inside torch.inference_mode(). In that context aten ignores ``tangent`` and
# ``level`` and returns a plain copy of the primal, which the candidate must
# reproduce: same shape/dtype, contiguous storage that does not alias the
# primal, neither input ever mutated, and no dual metadata attached to the
# result (unpack_dual reports a None tangent).
#
# The op performs no arithmetic, so element counts stay small (<= 96K) and
# ranks 0-5 are covered. Ranks are exercised through a shape parametrization;
# each (shape, dtype, level) triple is a distinct workload.
_MAKE_DUAL_COPY_SHAPES = [
    (),
    (1,),
    (64, 64),
    (20, 320, 15),
    (4, 8, 16, 32),
    (2, 3, 4, 5, 6),
]

# The op is a pure copy of the primal, so every dtype that aten can move is
# supported: all floats, all ints, bool, and complex64 (complex32 is excluded
# because the torch comparison helpers cannot materialize ComplexHalf on the
# reference device).
_MAKE_DUAL_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES + [torch.complex64]
)
_MAKE_DUAL_COPY_LEVELS = [0, 1]


def _make_input(shape, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_make_dual_copy" and "_make_dual_copy.out".
    return flag_gems.testing.resolve_gems_op(
        "_make_dual_copy", getattr(flag_gems, "_make_dual_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_make_dual_copy.out", getattr(flag_gems, "_make_dual_copy_out", None)
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp, tangent, ref_tangent, dtype):
    # _make_dual_copy returns a fresh contiguous copy: same shape/dtype, no
    # aliasing of the primal, and neither input is ever mutated.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    utils.gems_assert_equal(tangent, ref_tangent)
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_no_dual_tangent(res_out, ref_out):
    # The result is a plain tensor, not a dual: unpacking it must yield no
    # tangent. (The candidate cannot just wrap the primal in dual metadata.)
    _, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
    _, ref_tangent = torch.autograd.forward_ad.unpack_dual(ref_out)
    assert res_tangent is None
    assert ref_tangent is None


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("shape", _MAKE_DUAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
@pytest.mark.parametrize("level", _MAKE_DUAL_COPY_LEVELS)
def test__make_dual_copy(shape, dtype, level):
    with torch.inference_mode():
        inp = _make_input(shape, dtype)
        tangent = _make_input(shape, dtype)
        # Clone so the post-call equality checks below can detect any mutation
        # of the inputs even when the reference runs on the same device.
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_copy_semantics(
            res_out, ref_out, inp, ref_inp, tangent, ref_tangent, dtype
        )
        _assert_no_dual_tangent(res_out, ref_out)


@pytest.mark._make_dual_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
def test__make_dual_copy_out(shape, dtype):
    with torch.inference_mode():
        inp = _make_input(shape, dtype)
        tangent = _make_input(shape, dtype)
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())
        out = torch.empty_like(inp)
        ref_out = torch.empty_like(ref_inp)

        ref_ret = torch.ops.aten._make_dual_copy.out(
            ref_inp, ref_tangent, 0, out=ref_out
        )
        res_ret = _resolve_gems_op_out()(inp, tangent, 0, out=out)

        # The .out variant must write into and return the out tensor itself.
        assert res_ret is out
        assert ref_ret is ref_out
        _assert_copy_semantics(
            res_ret, ref_ret, inp, ref_inp, tangent, ref_tangent, dtype
        )
        utils.gems_assert_equal(out, ref_out)
        _assert_no_dual_tangent(res_ret, ref_ret)


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
def test__make_dual_copy_non_contiguous(shape, dtype):
    # The copy must read through arbitrary primal strides and still emit a
    # contiguous output. Slice on both the test device and the reference
    # device (inside inference_mode so both slices stay inference tensors).
    with torch.inference_mode():
        base = _make_input(shape, dtype)
        ref_base = utils.to_reference(base)
        inp = base[..., ::2]
        ref_inp = ref_base[..., ::2]
        tangent = _make_input(inp.shape, dtype)
        ref_tangent = utils.to_reference(tangent.clone())
        assert not inp.is_contiguous()

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, 0)
        res_out = _resolve_gems_op()(inp, tangent, 0)

        _assert_copy_semantics(
            res_out, ref_out, inp, ref_inp, tangent, ref_tangent, dtype
        )
        _assert_no_dual_tangent(res_out, ref_out)


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_copy_special_values(dtype):
    # A pure copy preserves every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    with torch.inference_mode():
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
            dtype=dtype,
            device=flag_gems.device,
        )
        tangent = torch.ones_like(values)
        ref_inp = utils.to_reference(values.clone())
        ref_tangent = utils.to_reference(tangent.clone())

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, 0)
        res_out = _resolve_gems_op()(values, tangent, 0)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()
