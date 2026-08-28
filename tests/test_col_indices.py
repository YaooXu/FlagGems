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

import itertools
import os
import sys

import pytest
import torch

import flag_gems

# KernelGen's verification harness stages the test files in a temporary
# copy of the FlagGems tree and runs pytest with --import-mode=importlib
# from a working directory that is not on sys.path, so the parent of this
# package may not be importable yet.  Make it importable before using the
# relative import below.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from . import accuracy_utils as utils  # noqa: E402

# aten::col_indices(Tensor(a) self) -> Tensor(a) returns the column index
# tensor of a sparse CSR tensor: shape batch_dims + (nnz,) with dtype int64.
# The result is an alias of the input's internal col_indices storage and is
# independent of the stored values, so every workload below feeds a sparse CSR
# tensor. Each (shape, nnz) pair is a distinct layout: 2-D all-sparse, 3-D
# batched, and 4-D multi-batch-dims, with varying nnz so the shape of the
# result is exercised.
_COL_CASES = [
    ((5, 4), 7),
    ((3, 8), 16),
    ((8, 3), 12),
    ((4, 4), 16),
    ((1, 6), 4),
    ((3, 5, 4), 7),
    ((2, 4, 6), 12),
    ((2, 3, 4, 5), 8),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSR runtime supports: every float, int, and bool
# family.
_COL_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype, seed=0):
    # Deterministic CPU-side generation, then the sparse CSR tensor is created
    # on the test device. (row, col) pairs are drawn with replacement:
    # duplicate entries are allowed and merely leave the tensor uncoalesced
    # (covered explicitly below). The crow pointer array is built with a
    # row-wise bincount, so it is always a valid CSR structure.
    gen = torch.Generator("cpu").manual_seed(seed)
    nrows, ncols = shape[-2], shape[-1]
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, generator=gen)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, generator=gen)
    order = torch.argsort(rows * ncols + cols, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    counts = torch.stack(
        [
            torch.bincount(rows[idx], minlength=nrows)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (nrows,))
    crow = torch.zeros(batch + (nrows + 1,), dtype=torch.long)
    crow[..., 1:] = torch.cumsum(counts, -1)
    if dtype.is_floating_point:
        values = torch.randn(entries_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, entries_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, entries_shape, dtype=dtype, generator=gen)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.col_indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems.col_indices callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "col_indices", getattr(flag_gems, "col_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # col_indices returns a fresh view of the input's internal
    # batch_dims + (nnz,) int64 column index tensor. The entries are exact, and
    # the schema annotation Tensor(a) self -> Tensor(a) requires the result to
    # alias the input's col_indices storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert ref_out.shape == inp.col_indices().shape
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal col_indices tensor.
    assert res_out.data_ptr() == torch.ops.aten.col_indices(inp).data_ptr()
    assert ref_out.data_ptr() == torch.ops.aten.col_indices(ref_inp).data_ptr()
    # The accessor must not mutate the input: the result still matches the
    # (untouched) col_indices captured on the reference copy before the call.
    utils.gems_assert_equal(res_out, torch.ops.aten.col_indices(ref_inp))


@pytest.mark.col_indices
@pytest.mark.parametrize("case", _COL_CASES)
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_empty(dtype):
    # nnz == 0: crow is all zeros, and col_indices must still return a
    # batch_dims + (0,) int64 tensor (not a dense or wrongly-shaped tensor).
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_single_row(dtype):
    # nrows == 1: crow has the degenerate shape (2,) with crow[0] == 0 and
    # crow[1] == nnz, and col_indices has shape (nnz,).
    shape, nnz = (1, 7), 5
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (cols[0] == cols[1] in row 0), which
    # leaves the tensor uncoalesced; col_indices must still return exactly the
    # stored col_indices tensor (never a coalesced/sorted copy). Row 0 holds 3
    # entries for columns [0, 0, 2], so a coalescing implementation would
    # visibly change the stored structure.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
