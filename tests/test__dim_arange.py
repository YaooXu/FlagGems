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

# ``_dim_arange`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._dim_arange`` and ``-m _dim_arange``
# both work.
setattr(
    pytest.mark,
    "_dim_arange",
    MarkDecorator(Mark("_dim_arange", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dim_arange(Tensor like, int dim) -> Tensor builds a fresh 1-D int64
# tensor of length like.size(dim) holding the values [0, 1, ..., like.size(dim)-1].
# Only the shape/device of ``like`` are consulted; its values and dtype never
# influence the result. The output is always a fresh (non-view, non-alias)
# int64 tensor. 0-D ``like`` raises IndexError for every dim, so it is excluded
# from the workloads below. These shapes mirror utils.POINTWISE_SHAPES.
_DIM_ARANGE_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [
        (1,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
)

# Every valid dim for every shape above. Positive and negative indexing are
# normalized identically by aten, so both conventions are exercised.
_DIM_ARANGE_CASES = [
    (shape, dim)
    for shape in _DIM_ARANGE_SHAPES
    for dim in range(-len(shape), len(shape))
]

# The op ignores the input values and dtype, so exercise every storage dtype
# family the runtime supports (float, int, and bool "like" tensors).
_DIM_ARANGE_INPUT_DTYPES = utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. ``flag_gems._dim_arange``
    # may not be registered yet, so getattr supplies a safe default and
    # resolve_gems_op falls back to the package namespace before raising.
    return flag_gems.testing.resolve_gems_op(
        "_dim_arange", getattr(flag_gems, "_dim_arange", None)
    )


@pytest.mark._dim_arange
@pytest.mark.parametrize("shape, dim", _DIM_ARANGE_CASES)
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = _resolve_gems_op()(inp, dim)

    # The result is a fresh 1-D int64 tensor, never a view/alias of ``like``.
    assert res_out.shape == ref_out.shape == (shape[dim],)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._dim_arange
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_non_contiguous(dtype):
    # _dim_arange must work on any tensor layout; only the shape is consulted.
    base = torch.zeros(4, 8, 6, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)  # (8, 4, 6), non-contiguous
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    assert res_out.shape == ref_out.shape == (8,)
    assert res_out.dtype == torch.int64
    assert not res_out._is_view()
    utils.gems_assert_equal(res_out, ref_out)
