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

# ``_dimI`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimI`` and ``-m _dimI`` both work.
setattr(
    pytest.mark,
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimI(Tensor self) -> int returns the number of sparse dimensions of a
# sparse tensor (``sparse_dim``): a pure metadata query whose result never
# depends on the index/data values or the storage dtype. Dense tensors have no
# Sparse* dispatch for this operator (they raise NotImplementedError), so every
# workload below feeds a sparse COO tensor. Each (shape, sparse_dim) pair is a
# distinct layout: 1-D all-sparse, 2-D/3-D all-sparse, and mixed sparse+dense
# ranks up to 5-D. Element counts stay small because the op only reads
# metadata.
_DIMI_CASES = [
    ((5,), 1),
    ((3, 4), 2),
    ((3, 4), 1),
    ((8, 8, 8), 3),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 1),
    ((3, 4, 5, 4, 5), 3),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse COO runtime supports: every float, int, and bool
# family.
_DIMI_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, sparse_dim, dtype, nnz=8, seed=0):
    # Deterministic CPU-side generation, then the sparse tensor is created on
    # the test device. Indices are drawn without replacement concerns: duplicate
    # indices are allowed and would merely leave the tensor uncoalesced, which
    # is covered separately below.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    if dtype.is_floating_point:
        values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._dimI is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimI callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op("_dimI", getattr(flag_gems, "_dimI", None))


def _assert_result(res_out, ref_out, sparse_dim):
    # _dimI returns a plain Python int holding the sparse dimension count, so
    # exact equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("case", _DIMI_CASES)
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI(case, dtype):
    shape, sparse_dim = case
    inp = _make_input(shape, sparse_dim, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty(dtype):
    # nnz == 0: indices and values are empty, but the sparse dims of the layout
    # are still reported exactly as for a populated tensor.
    shape, sparse_dim = (3, 4, 2), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 2, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _dimI must still report
    # the same sparse dim as the coalesced form (it never inspects the index or
    # data values). The (0, 1) coordinate is repeated three times.
    shape, sparse_dim = (3, 4), 2
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
