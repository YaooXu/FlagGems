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

# aten::_shape_as_tensor(Tensor self) -> Tensor materializes the logical shape
# of ``self`` as a fresh 1-D int64 tensor. Only the rank and sizes are
# consulted; the values, dtype, strides, and layout never influence the result,
# and 0-D inputs yield an empty 1-D int64 tensor. aten always builds the output
# on the CPU regardless of the input device, so the candidate must do the same.
# These shapes mirror utils.POINTWISE_SHAPES.
_SHAPE_AS_TENSOR_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [
        (),
        (1,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
)

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


def _assert_result(res_out, ref_out, shape):
    # The result is a fresh 1-D int64 CPU tensor holding the logical shape,
    # never a view/alias of the input.
    assert res_out.shape == ref_out.shape == (len(shape),)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == ref_out.device == torch.device("cpu")
    assert not res_out._is_view()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", _SHAPE_AS_TENSOR_SHAPES)
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_empty(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("case", _VIEW_CASES)
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_view(case, dtype):
    _, view_fn, expected = case
    base = torch.zeros(4, 8, 6, dtype=dtype, device=flag_gems.device)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)
