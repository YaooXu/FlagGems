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

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::sparse_bsc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse BSC tensor with the given compressed column pointers
# (length n_col_blocks + 1), stored row indices (length nnz) and block values
# (shape (nnz, Br, Bc)), laid out over a logical 2-D matrix of size ``size``.
# There is no .default overload (the size-less sibling
# ``aten::sparse_bsc_tensor.ccol_row_value`` infers the same shape whenever
# ``size`` equals the block-grid extent), so the reference always calls the
# schema above; the candidate is resolved by the same public operator name and
# invoked with exactly the same arguments (rule 6: torch_op and gems_op share
# call semantics).
#
# Construction copies the raw stored entries and index arrays verbatim (never
# coalescing duplicates or re-sorting rows), so the value comparisons below are
# bit-for-bit for every storage dtype. The op is a pure sparse factory: it
# performs no arithmetic on the values (nan/inf/-0.0 survive unchanged) and it
# is neither differentiable nor broadcastable, so the backward and broadcast
# dimensions of the regular-operator spec do not apply; the value-range, shape,
# nan/inf and negative dimensions are covered here instead.
#
# Dtype coverage (regular-operator spec): the required int8 / uint8 /
# float8_e4m3fn / float8_e5m2 / fp32 / bf16 / fp16 / int32 / int64 set, plus
# the shared fp64 / int16 sets and bool, probed at import time so a backend that
# cannot materialise a storage dtype is skipped cleanly (tu.supported_dtypes).
# Both supported index dtypes (int32, int64) are exercised for the structure.
_BSC_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)

# fp8 storage is validated with the exact (bit-for-bit) comparison paths only:
# torch.testing's tolerance-based comparison needs a CPU multiply that fp8 has
# no kernel for (RuntimeError "mul_cpu_reduced_float" not implemented for
# 'Float8_e5m2' -- it fails even for two identical fp8 tensors), and the sparse
# densification path has no fp8 kernel either (RuntimeError "index_add" not
# implemented for 'Float8_e4m3fn'). The factory copies the stored entries
# verbatim, so exact equality is the right check. Both fp8 dtypes do carry the
# special values (e5m2 stores +/-inf and nan bit-for-bit, e4m3fn preserves nan),
# so they are included in the nan/inf workload with equal_nan threaded through
# the exact comparisons below.
_BSC_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def _bsc_dtype_probe(op_name, dtype):
    """Report whether the sparse BSC factory accepts ``dtype`` storage."""
    del op_name
    try:
        ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
        row = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
        values = torch.zeros((2, 2, 2), dtype=dtype, device=flag_gems.device)
        out = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ccol,
            row,
            values,
            size=[4, 4],
            dtype=dtype,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
        return out.layout == torch.sparse_bsc and out.dtype == dtype
    except Exception:
        return False


# Probe the device before parametrizing: an op/dtype pair that cannot run must
# not be turned into a red test. If the probe yields nothing, keep the full
# candidate list rather than a float32-only fallback, so a failed/absent probe
# never silently drops the spec-required int8/uint8/fp8 dtypes.
_BSC_DTYPES = tu.supported_dtypes(
    "sparse_bsc_tensor", candidates=_BSC_DTYPE_CANDIDATES, probe=_bsc_dtype_probe
) or list(_BSC_DTYPE_CANDIDATES)
_BSC_FLOAT_DTYPES = [dtype for dtype in _BSC_DTYPES if dtype.is_floating_point]
# nan/inf/-inf are representable in every float family here, fp8 included (see
# the fp8 note above); the comparisons carry equal_nan where needed.
_BSC_NAN_INF_DTYPES = list(_BSC_FLOAT_DTYPES)
_INDEX_DTYPES = [torch.int32, torch.int64]

# (logical matrix shape, block size, nnz) structural cases: 2x2 row/col blocks
# with partial and full fill, larger blocks, a non-square matrix, 3x3 blocks,
# a single row block and the empty (nnz == 0) tensor. Every matrix dimension is
# an exact multiple of its block dimension.
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


def _bsc_shape_level_cases():
    """The shared tu.selected_shapes() levels mapped onto BSC layouts.

    A BSC tensor stores a 2-D logical matrix; any leading dimensions of the
    shared shapes are batch dims that this construction path (values of shape
    (nnz, Br, Bc), 2 sparse dims) does not carry, so only the trailing two dims
    are used. The block size is chosen as the largest power of two (up to 2)
    dividing each dimension, and nnz is capped so the large shared shapes stay
    cheap.
    """
    cases = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        if len(shape) < 2:
            continue
        nrows, ncols = shape[-2], shape[-1]
        block = (2 if nrows % 2 == 0 else 1, 2 if ncols % 2 == 0 else 1)
        n_blocks = (nrows // block[0]) * (ncols // block[1])
        nnz = min(6, max(1, n_blocks))
        cases.append(((nrows, ncols), block, nnz))
    if not cases:
        cases = [((4, 4), (2, 2), 4)]
    return cases


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
    if dtype in _BSC_FP8_DTYPES:
        # fp8 only supports the exact comparison (see the fp8 note at the top);
        # equal_nan is still honoured so the special-value workload can use fp8.
        utils.gems_assert_equal(res_out.values(), ref_out.values(), equal_nan=equal_nan)
        utils.gems_assert_equal(res_out, ref_out, equal_nan=equal_nan)
    elif dtype.is_floating_point:
        utils.gems_assert_close(
            res_out.values(), ref_out.values(), dtype, equal_nan=equal_nan
        )
        # Whole-tensor comparison covers layout, dtype, shape, indices, values.
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan)
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())
        utils.gems_assert_equal(res_out, ref_out)
    # The block values land at the (row block, col block) slots implied by
    # row_indices and ccol_indices, so the dense forms must match too.
    if check_dense and dtype not in _BSC_FP8_DTYPES:
        if dtype.is_floating_point:
            utils.gems_assert_close(
                res_out.to_dense(), ref_out.to_dense(), dtype, equal_nan=equal_nan
            )
        else:
            utils.gems_assert_equal(res_out.to_dense(), ref_out.to_dense())


def _assert_value_range_result(res_out, ref_out, dtype, *, equal_nan=False):
    _assert_result(res_out, ref_out, dtype, equal_nan=equal_nan)
    if dtype not in _BSC_FP8_DTYPES:
        # Value-range-friendly tolerance comparison (exact for int/bool,
        # rtol/atol for float, equal_nan in both cases). Sparse fp8 cannot be
        # compared by torch.testing, so it stays on the exact path above.
        tu.assert_result_close(res_out, ref_out)


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
@pytest.mark.parametrize("case", _bsc_shape_level_cases())
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_shape_levels(case, dtype, index_dtype):
    # The shared shape levels from the spec, mapped onto (nrows, ncols) BSC
    # layouts: the constructed tensor's logical size must equal the requested
    # shape and its (nnz, Br, Bc) values must round-trip verbatim.
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, ["-1", "1"], index_dtype=index_dtype
    )

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    assert tuple(res_out.shape) == tuple(shape)
    _assert_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_value_ranges(case, value_range, dtype):
    # Value-range sweep: construction copies the block values verbatim, so
    # every range (including the dtype-extreme [0, max] / [min, 0] ranges) must
    # round-trip exactly for every probed storage dtype.
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(shape, block, nnz, dtype, value_range)

    ref_out = _call_reference(ccol, row, values, shape, dtype)
    res_out = _call_candidate(ccol, row, values, shape, dtype)

    _assert_value_range_result(res_out, ref_out, dtype)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("dtype", _BSC_NAN_INF_DTYPES)
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

    _assert_value_range_result(res_out, ref_out, dtype, equal_nan=True)


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


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_non_tensor():
    # The aten schema requires a Tensor for ccol_indices; a Python scalar hits
    # the invalid-argument path and raises. The candidate must reject it too.
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = tu.make_input(torch.float32, (3, 2, 2), ["-1", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            3.14,
            utils.to_reference(row),
            utils.to_reference(values),
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(
            3.14,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
