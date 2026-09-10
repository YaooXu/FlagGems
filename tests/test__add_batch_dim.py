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
from torch._C._functorch import is_batchedtensor, is_legacy_batchedtensor

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``_add_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._add_batch_dim`` and ``-m
# _add_batch_dim`` both work.
setattr(
    pytest.mark,
    "_add_batch_dim",
    MarkDecorator(Mark("_add_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor is the
# functorch/vmap "wrap" primitive: it hides the physical dimension ``batch_dim``
# of ``self`` behind a lazy vmap batch dimension at nesting ``level``. The
# returned tensor is a zero-copy BatchedTensorImpl (legacy batched tensor): the
# storage is kept whole and the observable (logical) shape is ``self.shape``
# with ``batch_dim`` removed. Removing the hidden dim again with the matching
# level/batch_size/batch_dim reproduces the physical input exactly.
#
# The op performs no arithmetic (the result is a lazy metadata view), so every
# storage dtype is supported and the value-range tests only verify that
# unwrapping reproduces the exact stored values.
#
# Coverage map:
#   * shapes: the spec's seven shape levels minus the 0-dim scalar, which the op
#     rejects because there is no dimension to hide (covered as a negative case);
#     ranks 1-5, driven by ``pytest --quick`` through ``tu.selected_shapes()``;
#   * batch_dim: both ends (front / back) and the middle of the valid range;
#   * levels: vmap nesting levels 0, 1 and 3 (level is pure bookkeeping and must
#     not change the visible values);
#   * value ranges: all five spec ranges over every supported dtype, using the
#     shared ``tu.make_input`` helper;
#   * dtypes: fp16/bf16/fp32/fp64, int16/int32/int64, int8/uint8, float8_e4m3fn
#     /float8_e5m2 and bool (the op is a view, so all are accepted);
#   * non-contiguous inputs (strides and storage offset must survive);
#   * nan/inf/-inf and signed zeros must round-trip unchanged;
#   * negative: 0-dim input, negative ``level`` and non-tensor inputs are
#     rejected (matching aten's own validation).
#
# No broadcast dimension applies (the op is unary) and autograd does not run
# through the functorch batch wrapper (``torch.autograd.grad`` raises inside a
# BatchedTensor), so neither is tested here.

# The op is a pure metadata view, so it accepts every storage dtype. int8/uint8
# and the two float8 flavours are added on top of the shared dtype sets to meet
# the required dtype coverage.
_SPECIAL_VALUE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_ADD_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + _SPECIAL_VALUE_DTYPES
)

# float8 tensors cannot be fed directly to torch.testing.assert_close (and CUDA
# has no exp kernel for them), so comparisons upcast them losslessly to
# float32 first; the view is bit-exact, so the round-trip changes nothing.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# The spec's [-1, 0] range has no representable unsigned counterpart (it
# collapses to the empty [0, 0) interval), so it is skipped for uint8.
_UNSIGNED_INT_DTYPES = (torch.uint8,)


def _view_shapes():
    # _add_batch_dim needs an existing dimension to hide: rank >= 1. The shared
    # shape set contains a 0-dim scalar, which is filtered out here and covered
    # as an explicit negative case instead.
    return [shape for shape in tu.selected_shapes() if len(shape) >= 1]


def _batch_dims(shape):
    # Front, back and middle of the valid batch_dim range (deduplicated for
    # short shapes).
    return sorted({0, len(shape) // 2, len(shape) - 1})


def _add_batch_dim_cases():
    if tu.LEVEL == "quick":
        shapes = [(2, 19, 7)]
    else:
        shapes = _view_shapes()
    return [(shape, batch_dim) for shape in shapes for batch_dim in _batch_dims(shape)]


def _dtype_range_pairs():
    pairs = []
    for dtype in _ADD_BATCH_DIM_DTYPES:
        for value_range in tu.selected_ranges():
            if dtype in _UNSIGNED_INT_DTYPES and value_range == ["-1", "0"]:
                continue
            pairs.append((dtype, value_range))
    return pairs


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order is:
    # (1) override, (2) the direct flag_gems._add_batch_dim callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_add_batch_dim", getattr(flag_gems, "_add_batch_dim", None)
    )


def _as_comparable(t):
    if t.dtype in _FP8_DTYPES:
        return t.to(torch.float32)
    return t


def _assert_materialized_equal(res, ref):
    res = _as_comparable(res)
    ref = _as_comparable(ref)
    if res.dtype == torch.bool or not res.is_floating_point():
        utils.gems_assert_equal(res, ref)
    else:
        utils.gems_assert_close(res, ref, res.dtype, equal_nan=True)


def _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype):
    # The candidate must actually return a legacy BatchedTensorImpl, not a plain
    # logical view: on a non-batched tensor _remove_batch_dim falls back to
    # unsqueeze + expand, which can accidentally rebuild the input for some
    # (shape, batch_dim) combinations.
    assert is_legacy_batchedtensor(ref_out)
    assert is_legacy_batchedtensor(res_out)
    assert is_batchedtensor(res_out) == is_batchedtensor(ref_out)

    # Visible metadata must match aten exactly.
    assert res_out.dtype == ref_out.dtype == inp.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()

    # Unwrapping with the matching level/batch_size/batch_dim reproduces the
    # physical input; candidate and reference must agree exactly.
    batch_size = inp.size(batch_dim)
    ref_mat = torch.ops.aten._remove_batch_dim(ref_out, level, batch_size, batch_dim)
    res_mat = torch.ops.aten._remove_batch_dim(res_out, level, batch_size, batch_dim)
    _assert_materialized_equal(ref_mat, ref_inp)
    _assert_materialized_equal(res_mat, ref_mat)

    # Route an elementwise op through both batched views: the candidate's view
    # must expose the exact same logical elements as aten's. float8 is skipped
    # because CUDA has no exp kernel for it (the exact materialization above
    # already validates the view).
    if dtype.is_floating_point and dtype not in _FP8_DTYPES:
        ref_obs = torch.exp(ref_out)
        res_obs = torch.exp(res_out)
        ref_val = torch.ops.aten._remove_batch_dim(
            ref_obs, level, batch_size, batch_dim
        )
        res_val = torch.ops.aten._remove_batch_dim(
            res_obs, level, batch_size, batch_dim
        )
        _assert_materialized_equal(res_val, ref_val)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", _add_batch_dim_cases())
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim(shape, batch_dim, level, dtype):
    # [-1, 1] keeps every storage dtype valid (bool ignores the range); the
    # value-range sweep below covers all five spec ranges.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape", _view_shapes())
@pytest.mark.parametrize("dtype, value_range", _dtype_range_pairs())
def test__add_batch_dim_value_ranges(shape, dtype, value_range):
    # The lazy view must round-trip the exact stored values for every spec range
    # (int/bool bit-exact, floats with equal_nan=True).
    batch_dim = len(shape) // 2
    level = 0
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", [((8, 16, 32), 1), ((4, 8, 16, 32), 2)])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim_non_contiguous(shape, batch_dim, level, dtype):
    # The lazy view must preserve the strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__add_batch_dim_nan_inf(dtype):
    # The view never performs arithmetic, so nan/inf/-inf and signed zeros pass
    # through the lazy wrapper untouched. 1e30 also covers the overflow-to-inf
    # path in fp16/bf16.
    vals = [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        1.5,
        -2.5,
        1e30,
        -1e30,
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    batch_dim, level = 0, 0

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    assert is_legacy_batchedtensor(ref_out)
    assert is_legacy_batchedtensor(res_out)
    # The logical (visible) shape drops the hidden batch dim: for the 1-D input
    # below with batch_dim=0 the batched view exposes a 0-dim scalar.
    assert res_out.shape == ref_out.shape

    batch_size = inp.size(batch_dim)
    ref_mat = torch.ops.aten._remove_batch_dim(ref_out, level, batch_size, batch_dim)
    res_mat = torch.ops.aten._remove_batch_dim(res_out, level, batch_size, batch_dim)
    _assert_materialized_equal(ref_mat, ref_inp)
    _assert_materialized_equal(res_mat, ref_mat)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_0dim_input():
    # A 0-dim (scalar) input has no dimension to hide behind a vmap batch dim:
    # aten rejects it with RuntimeError and the candidate must too.
    inp = tu.make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 0, 0)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 0, 0)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_negative_level():
    # level must be non-negative: a vmap batch dim always has a nesting level
    # >= 0, and aten enforces this with an internal assert that surfaces as
    # RuntimeError. The candidate must reproduce the validation.
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 1, -1)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 1, -1)


@pytest.mark._add_batch_dim
def test__add_batch_dim_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar is rejected. The
    # candidate must fail too rather than silently wrapping a non-tensor.
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(3.14, 0, 0)
    with pytest.raises((TypeError, ValueError, AttributeError, RuntimeError)):
        _resolve_gems_op()(3.14, 0, 0)
