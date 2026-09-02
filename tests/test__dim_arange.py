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

import flag_gems

# ``_dim_arange`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._dim_arange`` and ``-m _dim_arange`` both
# work.
setattr(
    pytest.mark,
    "_dim_arange",
    MarkDecorator(Mark("_dim_arange", (), {}, _ispytest=True), _ispytest=True),
)

# The KernelGen verification harness stages this file in a temporary copy of the
# FlagGems tree and runs pytest in-process with ``--import-mode=importlib`` from
# that temp root, which is not placed on ``sys.path``. Bootstrap the checkout
# root from ``__file__`` so the relative imports below resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::_dim_arange(Tensor like, int dim) -> Tensor builds a fresh 1-D int64
# tensor of length like.size(dim) holding the values [0, 1, ..., like.size(dim)-1].
# Only the shape and device of ``like`` are consulted; its values, dtype, strides
# and layout never influence the result. The output is always a fresh (non-view,
# non-alias) int64 tensor on the same device as ``like``. 0-D ``like`` raises
# IndexError for every dim, so the scalar shape is excluded from the valid
# workloads below and covered by the negative tests instead.
#
# Shape coverage follows the regular-operator spec's level selection
# (quick/all via the pytest --quick flag): tu.selected_shapes() minus the
# 0-D scalar. Every valid dim is exercised in both the positive and negative
# indexing conventions, which aten normalizes identically.
#
# Adaptation notes for the regular-operator spec:
# - Broadcast: N/A -- the op takes a single ``like`` tensor.
# - Backward: N/A -- the output is an int64 index tensor with no autograd
#   support, so there is no gradient to compare.
# - Value ranges: the input values are semantically irrelevant, so the
#   value-range dimension below (tu.selected_ranges()) verifies that the
#   deterministic arange result is produced for every storage range.
# - nan/inf: covered by a dedicated case (non-finite storage values are
#   ignored, equal_nan semantics do not apply to the int64 output).
_DIM_ARANGE_SHAPES = tuple(s for s in tu.selected_shapes() if len(s) > 0)
_DIM_ARANGE_CASES = [
    (shape, dim)
    for shape in _DIM_ARANGE_SHAPES
    for dim in range(-len(shape), len(shape))
]

# The op ignores the input values and dtype, so exercise every storage dtype
# family the runtime supports (float, int, and bool "like" tensors).
_DIM_ARANGE_INPUT_DTYPES = utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Non-contiguous views of a (4, 8, 6) base: (view_fn, logical_shape, dim,
# expected_len). The logical shape, not the storage, must drive the result.
_VIEW_CASES = [
    (lambda b: b.transpose(0, 1), (8, 4, 6), 0, 8),
    (lambda b: b.transpose(0, 1), (8, 4, 6), 1, 4),
    (lambda b: b[0:3, 2:7, 1], (3, 5), 1, 5),
    (lambda b: b.narrow(1, 1, 5), (4, 5, 6), 1, 5),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. ``flag_gems._dim_arange``
    # may not be registered yet, so getattr supplies a safe default and
    # resolve_gems_op falls back to the package namespace before raising.
    return flag_gems.testing.resolve_gems_op(
        "_dim_arange", getattr(flag_gems, "_dim_arange", None)
    )


def _assert_arange_result(res_out, ref_out, inp, expected_len):
    # The result is a fresh 1-D int64 tensor on the ``like`` device holding
    # [0, ..., size(dim)-1]; it is never a view/alias of ``like``.
    assert res_out.shape == ref_out.shape == (expected_len,)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == inp.device
    assert ref_out.device == inp.device or ref_out.device == torch.device("cpu")
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._dim_arange
@pytest.mark.parametrize("shape, dim", _DIM_ARANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_value_ranges(shape, dim, value_range, dtype):
    # The result must be the deterministic arange(like.size(dim)) no matter
    # what values the storage holds, so every range from the regular-operator
    # spec is exercised here (this doubles as the value-range migration of the
    # original zeros-based workload).
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = _resolve_gems_op()(inp, dim)

    _assert_arange_result(res_out, ref_out, inp, shape[dim])


@pytest.mark._dim_arange
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_non_contiguous(view_case, value_range, dtype):
    # _dim_arange must work on any tensor layout; only the logical shape is
    # consulted, never the storage.
    view_fn, expected_shape, dim, expected_len = view_case
    base = tu.make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert inp.shape == expected_shape
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = _resolve_gems_op()(inp, dim)

    _assert_arange_result(res_out, ref_out, inp, expected_len)


@pytest.mark._dim_arange
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__dim_arange_nan_inf(dtype):
    # nan/inf are ordinary storage values for this op and must be ignored: the
    # result is still the deterministic arange sequence over the selected dim.
    inp = tu.make_input(dtype, (4, 8, 6), ["-1", "1"])
    inp = inp.clone()
    inp[0, :, 0] = float("inf")
    inp[1, :, 1] = float("-inf")
    inp[2, :, 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 1)
    res_out = _resolve_gems_op()(inp, 1)

    _assert_arange_result(res_out, ref_out, inp, 8)


@pytest.mark._dim_arange
def test__dim_arange_rejects_out_of_range_dim():
    # dim must satisfy -like.dim() <= dim < like.dim(); both the positive and
    # the negative out-of-range bounds must raise like aten does.
    inp = tu.make_input(torch.float32, (3, 5), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 2)
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, -3)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, 2)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, -3)


@pytest.mark._dim_arange
def test__dim_arange_rejects_zero_dim_like():
    # 0-D ``like`` has no dims to arange over; aten raises IndexError for any dim.
    inp = tu.make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 0)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, 0)


@pytest.mark._dim_arange
def test__dim_arange_rejects_non_integer_dim():
    # The schema requires an int ``dim``; a float is rejected at binding time.
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._dim_arange(inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, 1.5)
