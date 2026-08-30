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

import os
import sys

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator
from torch._C._functorch import is_batchedtensor, is_legacy_batchedtensor

import flag_gems

# The KernelGen harness runs pytest in-process with its own ``tests`` package
# (kernelgen/tests) earlier on sys.path than this checkout's ``tests`` package.
# With ``--import-mode=importlib`` pytest does not prepend the checkout root, so
# ``tests`` would resolve to the harness's package and ``from . import
# accuracy_utils`` would fail with ImportError during collection. Re-point the
# ``tests`` package at this file's directory before importing the helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests as _tests_pkg  # noqa: E402

if _HERE not in getattr(_tests_pkg, "__path__", []):
    sys.modules.pop("tests", None)
    import tests as _tests_pkg  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

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
# functorch/vmap primitive that wraps ``self`` in a legacy BatchedTensorImpl:
# the physical tensor is kept whole and ``batch_dim`` is hidden behind a lazy
# vmap batch dimension at ``level``, so the observable (logical) shape drops
# that dimension. The op is a pure zero-copy metadata view (no arithmetic ever
# runs through the lazy wrapper), so every storage dtype is supported and the
# value-range tests only need to verify that unwrapping reproduces the exact
# stored values. It requires an input of rank >= 1 (0-D inputs raise
# RuntimeError) and a non-negative ``level``; the candidate must reproduce
# both validations. Each (shape, batch_dim) pair below is a distinct
# parametrized workload; ranks 1-5 and both ends of the valid batch_dim range
# are covered. Element counts stay small (<= 96K) since the op only inspects
# metadata.
_ADD_BATCH_DIM_CASES = [
    ((16,), 0),
    ((64, 32), 0),
    ((64, 32), 1),
    ((2, 19, 7), 0),
    ((2, 19, 7), 1),
    ((2, 19, 7), 2),
    ((20, 320, 15), 1),
    ((4, 8, 16, 32), 2),
    ((4, 7, 5, 3, 6), 3),
]

# The op is a pure zero-copy view: no arithmetic happens at creation time, so
# every storage dtype is supported. The logical value is materialized (see
# ``_assert_batched_view``) before comparison, so floating-point dtypes still
# compare exactly.
_ADD_BATCH_DIM_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Representative rank-3 (shape, batch_dim, level) used by the value-range
# sweep. The lazy-view semantics do not depend on the shape, so a single
# mid-range batch_dim is enough to verify that every value range in the spec
# round-trips bit-for-bit.
_ADD_BATCH_DIM_VALUE_CASE = ((2, 19, 7), 1, 0)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order is:
    # (1) override, (2) the direct flag_gems._add_batch_dim callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_add_batch_dim", getattr(flag_gems, "_add_batch_dim", None)
    )


def _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype):
    # The op's whole purpose is to return a lazy BatchedTensorImpl, so the
    # candidate must produce one too. This is asserted explicitly because a
    # plain logical view would silently satisfy a naive round-trip for some
    # (shape, batch_dim) combinations: on a non-batched tensor _remove_batch_dim
    # falls back to unsqueeze + expand, which can accidentally rebuild the
    # input.
    assert is_legacy_batchedtensor(ref_out)
    assert is_legacy_batchedtensor(res_out)
    assert is_batchedtensor(res_out) == is_batchedtensor(ref_out)

    # Logical view metadata must match aten exactly: same dtype, shape and
    # storage layout of the visible (batch-dim-stripped) view.
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == inp.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()

    # Removing the hidden batch dim with the matching level, batch_dim and
    # batch_size reproduces the physical input; the candidate and the reference
    # must agree bit-for-bit (the materialization is a zero-copy view).
    batch_size = inp.size(batch_dim)
    ref_mat = torch.ops.aten._remove_batch_dim(ref_out, level, batch_size, batch_dim)
    res_mat = torch.ops.aten._remove_batch_dim(res_out, level, batch_size, batch_dim)
    utils.gems_assert_equal(ref_mat, ref_inp)
    utils.gems_assert_equal(res_mat, ref_mat)

    # For floating-point inputs, route an elementwise op through both batched
    # views: the candidate's view must expose the exact same logical elements
    # as aten's, so the materialized result equals exp() of the physical input.
    if dtype.is_floating_point:
        res_obs = torch.exp(res_out)
        ref_obs = torch.exp(ref_out)
        res_val = torch.ops.aten._remove_batch_dim(
            res_obs, level, batch_size, batch_dim
        )
        ref_val = torch.ops.aten._remove_batch_dim(
            ref_obs, level, batch_size, batch_dim
        )
        utils.gems_assert_close(res_val, ref_val, dtype)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", _ADD_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim(shape, batch_dim, level, dtype):
    # Values are irrelevant to the view itself (a representative [-1, 1] range
    # keeps every storage dtype valid); the dedicated value-range test below
    # sweeps the full spec ranges.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level, dtype)


@pytest.mark._add_batch_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim_value_ranges(value_range, dtype):
    # The lazy view must round-trip the exact stored values for every numeric
    # range in the spec: it exposes the same physical elements, so unwrapping
    # with the matching level/batch_dim/batch_size reproduces the input
    # bit-for-bit (tu.assert_result_close compares int/bool exactly and floats
    # with equal_nan=True).
    shape, batch_dim, level = _ADD_BATCH_DIM_VALUE_CASE
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = _resolve_gems_op()(inp, batch_dim, level)

    assert is_legacy_batchedtensor(ref_out)
    assert is_legacy_batchedtensor(res_out)
    assert res_out.dtype == ref_out.dtype == inp.dtype
    assert res_out.shape == ref_out.shape

    batch_size = inp.size(batch_dim)
    ref_mat = torch.ops.aten._remove_batch_dim(ref_out, level, batch_size, batch_dim)
    res_mat = torch.ops.aten._remove_batch_dim(res_out, level, batch_size, batch_dim)
    tu.assert_result_close(ref_mat, ref_inp)
    tu.assert_result_close(res_mat, ref_mat)


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
    # through the lazy wrapper untouched: unwrapping the batched view must
    # reproduce them exactly (tu.assert_result_close compares with
    # equal_nan=True). 1e30 also covers the overflow-to-inf path in fp16/bf16.
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
    # The logical (visible) shape drops the hidden batch dim: for the 1-D
    # input below with batch_dim=0 the batched view exposes a 0-dim scalar.
    assert res_out.shape == ref_out.shape

    batch_size = inp.size(batch_dim)
    ref_mat = torch.ops.aten._remove_batch_dim(ref_out, level, batch_size, batch_dim)
    res_mat = torch.ops.aten._remove_batch_dim(res_out, level, batch_size, batch_dim)
    tu.assert_result_close(ref_mat, ref_inp)
    tu.assert_result_close(res_mat, ref_mat)


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
