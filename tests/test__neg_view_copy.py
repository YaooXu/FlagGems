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

# aten::_neg_view_copy materializes the negative view as a fresh contiguous
# copy: the output must contain -self, must not alias the input, and must not
# mutate it. Negation flips the sign bit, so every dtype that aten::neg
# supports is exact; bool is rejected by aten::neg with a RuntimeError.
_NEG_VIEW_COPY_DTYPES = utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Keep the magnitude away from INT_MIN so negation cannot overflow (e.g.
    # int16 with -32768 wrapping back to itself).
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


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
    # aliasing of the input, and the input is never mutated.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    inp = _make_input(shape, dtype)
    # Clone so the post-call equality check below can detect any mutation of
    # the input even when the reference runs on the same device.
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._neg_view_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._neg_view_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)

    ref_ret = torch.ops.aten._neg_view_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark._neg_view_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
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
