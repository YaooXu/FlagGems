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

# aten::sparse_csc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse CSC tensor with the given compressed column pointers
# (length N + 1 for an N-column matrix), stored row indices (length nnz) and
# values (shape (nnz,) for 2-D, (batch..., nnz) for batched), laid out over a
# logical matrix of size ``size``. The no-size sibling
# (``aten::sparse_csc_tensor.ccol_row_value``) infers the shape from the index
# tensors: rows = max(row) + 1, cols = len(ccol) - 1.
#
# The reference and the candidate are called with the same keyword set. The
# ``dtype`` keyword is always passed explicitly: without it the aten op forces
# float32 storage ("dtype of values (...) must match dtype of sparse tensor").
# The ``device`` keyword is passed explicitly too: on CUDA this torch build
# fails to infer the device from the input tensors ("Values and compressed
# tensor instance need to be on the same device") unless the target device is
# given. ``layout=torch.sparse_csc`` pins the compressed-column layout.
_CSC_CASES = [
    ((4, 4), 3),  # 4x4, sparse fill
    ((8, 6), 5),  # non-square
    ((6, 8), 4),  # non-square, more columns than rows
    ((1, 5), 3),  # single row
    ((5, 1), 2),  # single column (ccol has exactly 2 entries)
    ((4, 4), 0),  # empty (nnz == 0)
]

# Batched CSC: the batch dims prefix the matrix, so ccol/row/values all gain
# the batch shape and the constructed tensor has shape (batch, M, N).
_CSC_BATCHED_CASES = [
    ((2, 4, 4), 4),
    ((3, 6, 5), 6),
]

# The no-size overload infers (rows, cols) = (max(row) + 1, len(ccol) - 1)
# from the index tensors; each case is (ccol list, row list, expected shape).
# The nnz entries may be uncoalesced or unsorted; the construction preserves
# the stored order verbatim.
_CSC_NO_SIZE_CASES = [
    ([0, 2, 3], [0, 1, 2], (3, 2)),
    ([0, 3, 3, 4], [1, 0, 2, 0], (3, 3)),
    ([0, 0, 2, 3], [0, 2, 1], (3, 3)),
    ([0, 4, 4], [1, 0, 2, 0], (3, 2)),
]

# The construction copies the stored entries and index arrays verbatim, so
# every value dtype the sparse CSC runtime supports (all floats, ints, and
# bool) and both supported index dtypes are exercised.
_CSC_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
_INDEX_DTYPES = [torch.int32, torch.int64]


def _make_values(nnz, dtype, shape=None, seed=0):
    # Deterministic CPU-side generation of the stored values; the tensor is
    # moved to the test device. Shape is (nnz,) or (batch..., nnz).
    gen = torch.Generator("cpu").manual_seed(seed)
    shape = (nnz,) if shape is None else shape
    if dtype.is_floating_point:
        values = torch.randn(shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(-5, 6, shape, dtype=dtype, generator=gen)
    return values.to(flag_gems.device)


def _make_csc_inputs(shape, nnz, dtype, seed=0, index_dtype=torch.int64):
    # Deterministic CPU-side generation of ccol_indices, row_indices and
    # values; the arrays are then moved to the test device. The nnz entries
    # are spread across the N column blocks by random cut points, and the row
    # indices are drawn with replacement (duplicates and unsorted rows are
    # legal CSC structure that the construction must keep verbatim).
    device = flag_gems.device
    gen = torch.Generator("cpu").manual_seed(seed)
    batch, (M, N) = shape[:-2], shape[-2:]
    total_batch = 1
    for d in batch:
        total_batch *= d
    if N <= 1:
        counts = torch.full((total_batch, 1), nnz, dtype=torch.long)
    else:
        cuts = torch.sort(
            torch.randint(0, nnz + 1, (total_batch, N - 1), generator=gen)
        ).values
        bounds = torch.cat(
            [
                torch.zeros(total_batch, 1, dtype=torch.long),
                cuts[:, : N - 1],
                torch.full((total_batch, 1), nnz, dtype=torch.long),
            ],
            dim=-1,
        )
        counts = bounds[..., 1:] - bounds[..., :-1]
    ccol = torch.cat(
        [torch.zeros(total_batch, 1, dtype=torch.long), torch.cumsum(counts, -1)],
        dim=-1,
    ).view(batch + (N + 1,))
    row = torch.randint(0, M, (total_batch, nnz), generator=gen).view(batch + (nnz,))
    values = _make_values(nnz, dtype, shape=batch + (nnz,), seed=seed + 1)
    return (
        ccol.to(device=device, dtype=index_dtype),
        row.to(device=device, dtype=index_dtype),
        values,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_csc_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_csc_tensor", getattr(flag_gems, "sparse_csc_tensor", None)
    )


def _assert_result(res_out, ref_out, dtype, index_dtype):
    # Construction semantics: a sparse CSC tensor with the requested layout,
    # storage dtype, logical shape, and the exact index structure, living on
    # the test device.
    assert res_out.layout == torch.sparse_csc
    assert ref_out.layout == torch.sparse_csc
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    # CSC has no dense (block) dims: for 2-D and batched construction alike
    # the batch dims are sparse batch dims, so dense_dim() is always 0.
    assert res_out.sparse_dim() == 2
    assert res_out.dense_dim() == 0
    assert ref_out.sparse_dim() == 2
    assert ref_out.dense_dim() == 0
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    # The stored structure is transferred verbatim: compressed column
    # pointers, row indices and values all match the reference exactly (never
    # re-sorted or coalesced).
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())
    # Whole-tensor comparison covers layout, dtype, shape, indices and values.
    if dtype in utils.ALL_INT_DTYPES + utils.BOOL_TYPES:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
    # The stored entries land at the (row, col) slots implied by row_indices
    # and ccol_indices, so the dense forms must match too.
    if dtype in utils.ALL_INT_DTYPES + utils.BOOL_TYPES:
        utils.gems_assert_equal(res_out.to_dense(), ref_out.to_dense())
    else:
        utils.gems_assert_close(res_out.to_dense(), ref_out.to_dense(), dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("case", _CSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor(case, dtype, index_dtype):
    shape, nnz = case
    ccol, row, values = _make_csc_inputs(shape, nnz, dtype, index_dtype=index_dtype)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("case", _CSC_BATCHED_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_batched(case, dtype, index_dtype):
    shape, nnz = case
    ccol, row, values = _make_csc_inputs(shape, nnz, dtype, index_dtype=index_dtype)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("case", _CSC_NO_SIZE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_no_size(case, dtype, index_dtype):
    # Size-inferred overload: the 3-argument call (no size) derives the tensor
    # size from the index tensors: rows = max(row) + 1, cols = len(ccol) - 1.
    ccol_list, row_list, expected_shape = case
    ccol = torch.tensor(ccol_list, dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor(row_list, dtype=index_dtype, device=flag_gems.device)
    values = _make_values(len(row_list), dtype)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    assert tuple(res_out.shape) == expected_shape
    _assert_result(res_out, ref_out, dtype, index_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_uncoalesced(dtype, index_dtype):
    # The (row 0, col 0) slot is stored twice (row_indices[0] ==
    # row_indices[1] inside column 0), so the source is uncoalesced; the
    # construction must transfer the duplicate entries verbatim into the
    # tensor (never coalesce them) and the dense form accumulates both
    # entries.
    ccol = torch.tensor([0, 2, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [4, 3],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        [4, 3],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_unsorted_rows(dtype, index_dtype):
    # Row indices inside a column are deliberately not sorted (1, 0, 2 in
    # column 0); the construction must preserve the stored order instead of
    # re-sorting the entries.
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 2], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, seed=1)
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [4, 3],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _resolve_gems_op()(
        ccol,
        row,
        values,
        [4, 3],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
