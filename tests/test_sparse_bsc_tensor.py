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

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

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
#
# Construction copies the raw stored entries and index arrays verbatim (never
# coalescing duplicates or re-sorting rows), so the value comparisons below are
# bit-for-bit for every storage dtype. The op is a pure sparse factory: it
# performs no arithmetic on the values (nan/inf/-0.0 survive unchanged) and it
# is neither differentiable nor broadcastable, so the backward and broadcast
# dimensions of the regular-operator spec do not apply; the value-range, shape,
# nan/inf and negative dimensions are covered here instead.
#
# Dtype coverage: every storage dtype the sparse BSC runtime accepts (all
# floats, all ints and bool) and both supported index dtypes are exercised.
_BSC_CASES = [
    ((4, 4), (2, 2), 3),  # 2x2 row/col blocks, partial fill
    ((8, 8), (2, 2), 16),  # full 4x4 block grid
    ((16, 16), (4, 4), 8),  # larger blocks
    ((6, 8), (2, 2), 6),  # non-square matrix
    ((6, 6), (3, 3), 4),  # 3x3 blocks
    ((2, 6), (2, 3), 2),  # single row block
    ((4, 4), (2, 2), 0),  # empty (nnz == 0)
]

# Value-range sweep subset: small enough to keep the parametrization count
# bounded while covering a partial block-grid fill and a non-square matrix.
_BSC_VALUE_CASES = [
    ((4, 4), (2, 2), 3),
    ((6, 8), (2, 2), 6),
]

# Legacy storage: values of shape (nnz,) are 1x1 blocks. The reference accepts
# this layout but cannot densify it (to_dense() fails), so these workloads
# assert the constructed structure only.
_LEGACY_CASES = [
    ((4, 5), 3),
    ((4, 5), 0),
]

_BSC_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_BSC_INT_DTYPES = utils.ALL_INT_DTYPES
_BSC_DTYPES = _BSC_FLOAT_DTYPES + _BSC_INT_DTYPES + utils.BOOL_TYPES
_INDEX_DTYPES = [torch.int32, torch.int64]

# Integer/bool value ranges: one negative, one positive plus the dtype
# extremes. The full selected_ranges() sweep is reserved for floats (the
# non-extreme float ranges are meaningless for the exact integer path).
_INT_VALUE_RANGES = [
    ["-1", "1"],
    ["min", "0"],
    ["0", "max"],
]


def _make_bsc_structure(shape, block, nnz, seed=0, index_dtype=torch.int64):
    # Deterministic CPU-side generation of ccol_indices and row_indices: the
    # nnz entries are spread across the n_col_blocks column blocks by random
    # cut points, and the row indices are drawn with replacement (duplicates
    # and unsorted rows are legal BSC structure that the construction must
    # keep verbatim).
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
    return ccol.to(flag_gems.device), row.to(flag_gems.device)


def _make_bsc_inputs(
    shape, block, nnz, dtype, value_range, seed=0, index_dtype=torch.int64
):
    # Block values come from the shared value-range framework (tu.make_input):
    # range-bound symbols resolve per-dtype, so every storage dtype gets valid
    # inputs within the requested numeric range.
    ccol, row = _make_bsc_structure(
        shape, block, nnz, seed=seed, index_dtype=index_dtype
    )
    values = tu.make_input(dtype, (nnz,) + tuple(block), value_range).to(
        flag_gems.device
    )
    return ccol, row, values


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_bsc_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_bsc_tensor", getattr(flag_gems, "sparse_bsc_tensor", None)
    )


def _call_reference(ccol, row, values, size, dtype):
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)
    return torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=list(size),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )


def _call_candidate(ccol, row, values, size, dtype):
    return _resolve_gems_op()(
        ccol,
        row,
        values,
        size=list(size),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )


def _assert_result(res_out, ref_out, dtype, *, check_dense=True, equal_nan=False):
    # Construction semantics: a sparse BSC tensor with exact rank 2 sparse
    # dims, zero dense dims, the requested storage dtype and the requested
    # logical size.
    assert res_out.layout == torch.sparse_bsc
    assert ref_out.layout == torch.sparse_bsc
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert res_out.sparse_dim() == 2
    assert ref_out.sparse_dim() == 2
    assert res_out.dense_dim() == ref_out.dense_dim()
    if check_dense:
        # The standard (nnz, Br, Bc) block-values layout carries no dense dims;
        # the legacy 1D-values layout reports dense_dim == -2 and is only
        # structure-checked (check_dense=False).
        assert res_out.dense_dim() == 0
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    # The stored structure is transferred verbatim: compressed column
    # pointers, row indices and block values all match the reference exactly
    # (never re-sorted or coalesced).
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    if dtype.is_floating_point:
        utils.gems_assert_close(
            res_out.values(), ref_out.values(), dtype, equal_nan=equal_nan
        )
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())
    # Whole-tensor comparison covers layout, dtype, shape, indices and values.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    # The block values land at the (row block, col block) slots implied by
    # row_indices and ccol_indices, so the dense forms must match too.
    if check_dense:
        if dtype.is_floating_point:
            utils.gems_assert_close(
                res_out.to_dense(), ref_out.to_dense(), dtype, equal_nan=equal_nan
            )
        else:
            utils.gems_assert_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor(case, dtype, index_dtype):
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, ["-1", "1"], index_dtype=index_dtype
    )

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _BSC_FLOAT_DTYPES)
def test_sparse_bsc_tensor_float_value_ranges(case, value_range, dtype):
    # Value-range sweep over the full float family: construction copies the
    # block values verbatim, so every range (including the dtype-extreme
    # [0, max] / [min, 0] ranges) must round-trip exactly.
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(shape, block, nnz, dtype, value_range)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_VALUE_CASES)
@pytest.mark.parametrize("value_range", _INT_VALUE_RANGES)
@pytest.mark.parametrize("dtype", _BSC_INT_DTYPES + utils.BOOL_TYPES)
def test_sparse_bsc_tensor_int_value_ranges(case, value_range, dtype):
    # int/bool abs-exact path: the extreme [min, 0] / [0, max] ranges hit the
    # full integer span (int16 min through int64 max) with no wrap-around.
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(shape, block, nnz, dtype, value_range)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_result(res_out, ref_out, dtype)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("dtype", _BSC_FLOAT_DTYPES)
def test_sparse_bsc_tensor_nan_inf(dtype):
    # The factory copies the raw block values and performs no arithmetic on
    # them, so inf/-inf/nan/-0.0 survive the construction unchanged (and
    # 1e30/-1e30 cover the overflow-to-inf path in fp16/bf16). equal_nan
    # tolerates the nan outputs in every comparison below.
    values = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
            float("-inf"),
            float("inf"),
            float("nan"),
            -1.5,
            2.5,
            0.0,
            -0.0,
            -1e30,
            1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    ).reshape(2, 3, 3)
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [6, 6], dtype)
    res_out = _call_candidate(ccol, row, values, [6, 6], dtype)

    _assert_result(res_out, ref_out, dtype, equal_nan=True)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _LEGACY_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_legacy(case, dtype, index_dtype):
    # Legacy storage: values has shape (nnz,) instead of (nnz, Br, Bc). The
    # reference cannot densify such tensors, so the workload asserts the
    # constructed structure only (layout, logical size, dtype, and verbatim
    # ccol/row/values).
    shape, nnz = case
    ccol, row = _make_bsc_structure(shape, (1, 1), nnz, index_dtype=index_dtype)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

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
    values = tu.make_input(dtype, (3, 2, 2), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [4, 4], dtype)
    res_out = _call_candidate(ccol, row, values, [4, 4], dtype)

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
    values = tu.make_input(dtype, (3, 2, 2), ["-1", "1"]).to(flag_gems.device)

    ref_out = _call_reference(ccol, row, values, [4, 4], dtype)
    res_out = _call_candidate(ccol, row, values, [4, 4], dtype)

    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_dtype_mismatch():
    # The values tensor dtype must match the requested sparse tensor dtype;
    # the reference raises RuntimeError and the candidate must fail too.
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float64, (3, 2, 2), ["-1", "1"])

    with pytest.raises(RuntimeError):
        _call_reference(ccol, row, values, [4, 4], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(ccol, row, values, [4, 4], torch.float32)


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_layout():
    # Only the sparse_bsc layout is accepted; any other layout raises.
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(
            ccol,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_size():
    # A negative logical size is rejected (numel overflow); the candidate must
    # fail too rather than accept a nonsensical shape.
    ccol = torch.tensor([0, 2, 3], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])
    ref_ccol = utils.to_reference(ccol)
    ref_row = utils.to_reference(row)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(
            ccol,
            row,
            values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
