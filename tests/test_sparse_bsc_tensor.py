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

# aten::sparse_bsc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse BSC tensor with the given compressed column pointers
# (length n_col_blocks + 1), stored row indices (length nnz) and block values
# (shape (nnz, Br, Bc)), laid out over a logical matrix of size ``size``. There
# is no .default overload (the size-less sibling
# ``aten::sparse_bsc_tensor.ccol_row_value`` infers the same shape whenever
# ``size`` equals the block-grid extent), so the reference always calls the
# schema above; the candidate is resolved by the same public operator name and
# invoked with exactly the same arguments.
_BSC_CASES = [
    ((4, 4), (2, 2), 3),  # 2x2 row/col blocks, partial fill
    ((8, 8), (2, 2), 16),  # full 4x4 block grid
    ((16, 16), (4, 4), 8),  # larger blocks
    ((6, 8), (2, 2), 6),  # non-square matrix
    ((6, 6), (3, 3), 4),  # 3x3 blocks
    ((2, 6), (2, 3), 2),  # single row block
    ((4, 4), (2, 2), 0),  # empty (nnz == 0)
]

# Legacy storage: values of shape (nnz,) are 1x1 blocks. The reference accepts
# this layout but cannot densify it (to_dense() fails), so these workloads
# assert the constructed structure only.
_LEGACY_CASES = [
    ((4, 5), 3),
    ((4, 5), 0),
]

# The construction copies raw stored entries and index arrays, so every value
# dtype the sparse BSC runtime supports (all floats, ints, and bool) and both
# supported index dtypes are exercised.
_BSC_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
_INDEX_DTYPES = [torch.int32, torch.int64]


def _make_bsc_inputs(shape, block, nnz, dtype, seed=0, index_dtype=torch.int64):
    # Deterministic CPU-side generation of ccol_indices, row_indices and
    # values; the arrays are then moved to the test device. The nnz entries
    # are spread across the n_col_blocks column blocks by random cut points,
    # and the row indices are drawn with replacement (duplicates and unsorted
    # rows are legal BSC structure that the construction must keep verbatim).
    gen = torch.Generator("cpu").manual_seed(seed)
    M, N = shape
    Br, Bc = block
    n_row_blocks = M // Br
    n_col_blocks = N // Bc
    if n_col_blocks <= 1:
        counts = torch.full((n_col_blocks,), nnz, dtype=torch.long)
    else:
        cuts = torch.sort(
            torch.randint(0, nnz + 1, (n_col_blocks - 1,), generator=gen)
        ).values
        bounds = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                cuts,
                torch.full((1,), nnz, dtype=torch.long),
            ]
        )
        counts = bounds[1:] - bounds[:-1]
    ccol = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)]).to(
        index_dtype
    )
    row = torch.randint(0, n_row_blocks, (nnz,), generator=gen).to(index_dtype)
    values_shape = (nnz,) + tuple(block)
    if dtype.is_floating_point:
        values = torch.randn(values_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)
    return (
        ccol.to(flag_gems.device),
        row.to(flag_gems.device),
        values.to(flag_gems.device),
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_bsc_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_bsc_tensor", getattr(flag_gems, "sparse_bsc_tensor", None)
    )


def _assert_result(res_out, ref_out, dtype, *, check_dense=True):
    # Construction semantics: a sparse BSC tensor with exact rank 2 sparse
    # dims, zero dense dims, the requested storage dtype and the requested
    # logical size.
    assert res_out.layout == torch.sparse_bsc
    assert ref_out.layout == torch.sparse_bsc
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert res_out.sparse_dim() == 2
    assert res_out.dense_dim() == 0
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    # The stored structure is transferred verbatim: compressed column
    # pointers, row indices and block values all match the reference exactly
    # (never re-sorted or coalesced).
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())
    # Whole-tensor comparison covers layout, dtype, shape, indices and values.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    # The block values land at the (row block, col block) slots implied by
    # row_indices and ccol_indices, so the dense forms must match too.
    if check_dense:
        if dtype.is_floating_point:
            utils.gems_assert_close(res_out.to_dense(), ref_out.to_dense(), dtype)
        else:
            utils.gems_assert_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor(case, dtype, index_dtype):
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, index_dtype=index_dtype
    )
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=list(shape),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        size=list(shape),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _LEGACY_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_legacy(case, dtype, index_dtype):
    # Legacy storage: values has shape (nnz,), i.e. 1x1 blocks. The reference
    # cannot densify such tensors, so the workload asserts the constructed
    # structure only (layout, logical size, dtype, and verbatim ccol/row/
    # values).
    shape, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, (1, 1), nnz, dtype, index_dtype=index_dtype
    )
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=list(shape),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        size=list(shape),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, check_dense=False)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_uncoalesced(dtype, index_dtype):
    # The (row block 0, col block 0) slot is stored twice (row_indices[0] ==
    # row_indices[1] inside column block 0), so the source is uncoalesced; the
    # construction must transfer the duplicate entries verbatim into the
    # tensor (never coalesce them) and the dense form accumulates both blocks.
    ccol = torch.tensor([0, 2, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=index_dtype, device=flag_gems.device)
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((3, 2, 2), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (3, 2, 2), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (3, 2, 2), dtype=dtype, generator=gen)
    values = values.to(flag_gems.device)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=[4, 4],
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        size=[4, 4],
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unsorted_rows(dtype, index_dtype):
    # Row indices inside a column block are deliberately not sorted
    # (1, 0, 1 in column block 0); the construction must preserve the stored
    # order instead of re-sorting the entries.
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 1], dtype=index_dtype, device=flag_gems.device)
    gen = torch.Generator("cpu").manual_seed(1)
    if dtype.is_floating_point:
        values = torch.randn((3, 2, 2), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (3, 2, 2), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (3, 2, 2), dtype=dtype, generator=gen)
    values = values.to(flag_gems.device)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=[4, 4],
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        size=[4, 4],
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype)
