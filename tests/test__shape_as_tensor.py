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

# ``_shape_as_tensor`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._shape_as_tensor`` and ``-m
# _shape_as_tensor`` both work.
setattr(
    pytest.mark,
    "_shape_as_tensor",
    MarkDecorator(Mark("_shape_as_tensor", (), {}, _ispytest=True), _ispytest=True),
)

# The KernelGen verification harness stages this file in a temporary copy of the
# FlagGems tree and runs pytest in-process with ``--import-mode=importlib`` from
# that temp root, which is not placed on ``sys.path``. Bootstrap the checkout
# root from ``__file__`` so the relative import below resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::_shape_as_tensor(Tensor self) -> Tensor materializes the logical shape
# of ``self`` as a fresh 1-D int64 tensor. Only the rank and sizes are
# consulted; the values, dtype, strides, and layout never influence the result,
# and 0-D inputs yield an empty 1-D int64 tensor. aten always builds the output
# on the CPU regardless of the input device, so the candidate must do the same.
#
# Shape coverage follows the regular-operator spec's level selection
# (quick/all via the pytest --quick flag): tu.selected_shapes(), which
# includes the 0-D scalar (mapped to the empty 1-D output) and ranks up to 8.
#
# Adaptation notes for the regular-operator spec:
# - Broadcast: N/A -- the op takes a single ``self`` tensor.
# - Backward: N/A -- the output is a fresh int64 metadata tensor with no
#   autograd support, so there is no gradient to compare.
# - Value ranges: the input values are semantically irrelevant, so the
#   value-range dimension below (tu.selected_ranges()) verifies that the
#   deterministic shape materialization is produced for every storage range.
# - nan/inf: covered by a dedicated case (non-finite storage values are
#   ignored; the int64 output compares exactly).

# The op ignores the input values and dtype, so exercise every storage dtype
# family the runtime supports (float, int, and bool "like" tensors).
_SHAPE_AS_TENSOR_INPUT_DTYPES = (
    utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

# Zero-size dimensions are part of the logical shape and must be reported
# faithfully; a ``numel == 0`` fast path would silently drop them.
_EMPTY_SHAPES = [(0,), (0, 5), (3, 0, 4)]


def _transposed_view(base):
    return base.transpose(0, 1)


def _sliced_view(base):
    return base[0:3, 2:7, 1]


def _narrowed_view(base):
    return base.narrow(1, 1, 5)


# Each case builds a non-contiguous view of a (4, 8, 6) base and states the
# logical shape the query must report. A transposed, sliced, or narrowed view
# changes the sizes, so a candidate that reads the base storage shape instead
# of the view's logical shape would fail these checks.
_VIEW_CASES = [
    ("transposed", _transposed_view, (8, 4, 6)),
    ("sliced", _sliced_view, (3, 5)),
    ("narrowed", _narrowed_view, (4, 5, 6)),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins.
    # ``flag_gems._shape_as_tensor`` may not be registered yet, so getattr
    # supplies a safe default and resolve_gems_op falls back to the package
    # namespace before raising.
    return flag_gems.testing.resolve_gems_op(
        "_shape_as_tensor", getattr(flag_gems, "_shape_as_tensor", None)
    )


def _assert_result(res_out, ref_out, inp, shape):
    # The result is a fresh 1-D int64 CPU tensor holding the logical shape,
    # never a view/alias of the input.
    assert res_out.shape == ref_out.shape == (len(shape),)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == ref_out.device == torch.device("cpu")
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_value_ranges(shape, value_range, dtype):
    # The result must be the deterministic shape materialization no matter what
    # values the storage holds, so every range from the regular-operator spec
    # is exercised here (this doubles as the value-range migration of the
    # original zeros-based workload).
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_empty(shape, value_range, dtype):
    # Zero-size dimensions are part of the logical shape; a ``numel == 0`` fast
    # path that drops them would fail here.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_non_contiguous(view_case, value_range, dtype):
    # _shape_as_tensor must work on any tensor layout; only the logical shape
    # is consulted, never the storage.
    _, view_fn, expected = view_case
    base = tu.make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert inp.shape == expected
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, expected)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__shape_as_tensor_nan_inf(dtype):
    # nan/inf are ordinary storage values for this op and must be ignored: the
    # result is still the deterministic shape tensor over the logical shape.
    inp = tu.make_input(dtype, (4, 8, 6), ["-1", "1"])
    inp = inp.clone()
    inp[0, :, 0] = float("inf")
    inp[1, :, 1] = float("-inf")
    inp[2, :, 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, (4, 8, 6))


@pytest.mark._shape_as_tensor
def test__shape_as_tensor_rejects_non_tensor_input():
    # The schema requires a Tensor ``self``; every non-tensor argument is
    # rejected at binding time by aten, and the candidate must not silently
    # accept it either.
    for bad in (5, [1, 2, 3], "abc"):
        with pytest.raises(RuntimeError):
            torch.ops.aten._shape_as_tensor(bad)
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(bad)
