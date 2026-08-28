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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate
# a marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimV(Tensor self) -> int is a sparse-only metadata query that returns
# the number of dense dimensions (``self.dense_dim()``) of a sparse tensor. It
# is only dispatched on the sparse backends (SparseCPU/SparseCUDA/SparseMeta/
# SparseXPU), so every input below is a sparse COO tensor. Each case is a
# (sparse_shape, dense_shape, nnz) triple: the tensor size is
# ``sparse_shape + dense_shape`` and the expected result is ``len(dense_shape)``.
# The pairs cover all-sparse layouts (dense_dim == 0) as well as mixed
# sparse+dense ranks with one, two and three dense dimensions.
_DIMV_CASES = [
    ((4, 4), (), 8),
    ((8, 8, 8), (), 64),
    ((4, 4), (3,), 8),
    ((2, 3, 4), (5,), 12),
    ((16, 16), (7, 13), 40),
    ((2, 3, 4), (5, 6), 12),
    ((3,), (4, 5, 6), 2),
]

# The result does not depend on the stored values, but the candidate path must
# accept every storage dtype the sparse COO runtime supports: all float, int
# and bool families.
_DIMV_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_sparse_coo(case, dtype, device):
    # Deterministic CPU-side generation; the sparse tensor is then constructed
    # on the requested device. Duplicate indices are allowed (the layout is
    # simply uncoalesced), which is covered separately below.
    sparse_shape, dense_shape, nnz = case
    gen = torch.Generator("cpu").manual_seed(2026)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values_shape = (nnz,) + tuple(dense_shape)
    if dtype.is_floating_point:
        values = torch.rand(values_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._dimV is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimV callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op("_dimV", getattr(flag_gems, "_dimV", None))


def _assert_result(res_out, ref_out, dense_dim):
    # _dimV returns a plain Python int holding the dense dimension count, so
    # exact equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == dense_dim


@pytest.mark._dimV
@pytest.mark.parametrize("case", _DIMV_CASES)
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV(case, dtype):
    inp = _make_sparse_coo(case, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(case[1]))
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == len(case[1])
    assert inp.sparse_dim() == len(case[0])


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty(dtype):
    # nnz == 0: indices and values are empty, but the dense dims of the layout
    # are still reported exactly as for a populated tensor.
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(dense_shape))


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_uncoalesced(dtype):
    # The (0, 0) coordinate is repeated, so the tensor is uncoalesced; _dimV
    # must still report the same dense dims as the coalesced form because it
    # never inspects the index or data values.
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    gen = torch.Generator("cpu").manual_seed(0)
    values_shape = (5,) + tuple(dense_shape)
    if dtype.is_floating_point:
        values = torch.randn(values_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(dense_shape))
