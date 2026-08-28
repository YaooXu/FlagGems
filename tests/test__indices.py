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

# ``_indices`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._indices`` and ``-m _indices`` both
# work.
setattr(
    pytest.mark,
    "_indices",
    MarkDecorator(Mark("_indices", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_indices(Tensor(a) self) -> Tensor(a) returns the (sparse_dim, nnz)
# int64 index tensor of a sparse COO tensor. The result is an alias of the
# input's internal indices storage and never depends on the stored values, so
# every workload below feeds a sparse COO tensor. Each (shape, sparse_dim, nnz)
# triple is a distinct layout: 1-D all-sparse, 2-D/3-D all-sparse, and mixed
# sparse+dense ranks up to 5-D, with varying nnz so the (sparse_dim, nnz) shape
# of the result is exercised.
_INDICES_CASES = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse COO runtime supports: every float, int, and bool
# family.
_INDICES_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, sparse_dim, nnz, dtype, seed=0):
    # Deterministic CPU-side generation, then the sparse tensor is created on
    # the test device. Indices are drawn with replacement: duplicate indices
    # are allowed and merely leave the tensor uncoalesced (covered explicitly
    # below).
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
    # default stays None until flag_gems._indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._indices callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_indices", getattr(flag_gems, "_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # _indices returns a fresh view of the input's internal (sparse_dim, nnz)
    # int64 index tensor. The values are exact, and the schema annotation
    # Tensor(a) self -> Tensor(a) requires the result to alias the input's
    # indices storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert res_out.shape == (inp.sparse_dim(), inp._nnz())
    assert ref_out.shape == (ref_inp.sparse_dim(), ref_inp._nnz())
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal indices tensor.
    assert res_out.data_ptr() == inp._indices().data_ptr()
    assert ref_out.data_ptr() == ref_inp._indices().data_ptr()
    # The accessor must not mutate the input: the result still matches the
    # (untouched) indices captured on the reference copy before the call.
    utils.gems_assert_equal(res_out, ref_inp._indices())


@pytest.mark._indices
@pytest.mark.parametrize("case", _INDICES_CASES)
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty(dtype):
    # nnz == 0: indices and values are empty, but _indices must still return a
    # (sparse_dim, 0) int64 tensor (not a dense or wrongly-shaped tensor).
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _indices must still
    # return exactly the stored index tensor (never a coalesced/sorted copy).
    # The (0, 1) coordinate is repeated three times and the entries are NOT
    # sorted, so a coalescing implementation would visibly change the result.
    shape = (3, 4)
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

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
