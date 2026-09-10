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

# aten::col_indices(Tensor(a) self) -> Tensor(a) returns the column index
# tensor of a sparse row-compressed tensor (CSR or BSR): shape batch_dims +
# (nnz,) with dtype int64. The result is a view of the input's internal
# col_indices storage and never depends on the stored values, so every workload
# below feeds a sparse row-compressed tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (layout, shape, nnz, blocks) layouts from the quick/all
#     levels, ranks 2-7 (2-D all-sparse, 3-D/4-D batched, higher-rank
#     multi-batch-dims, and BSR with varied block shapes), with varying nnz so
#     the batch_dims + (nnz,) shape of the result is exercised;
#   * value ranges: tu.selected_ranges() (the five spec ranges) over
#     representative layouts, so every supported storage dtype is exercised
#     with negative, positive, extreme and degenerate value ranges (the
#     returned col_indices is identical for all of them);
#   * dtype coverage: every storage dtype the sparse row-compressed runtime
#     accepts, probed on the active device -- int8/uint8/fp8/fp16/bf16/fp32/
#     fp64/int16/int32/int64/bool are all accepted because the accessor never
#     reads the values payload;
#   * edge cases: empty (nnz == 0, unbatched and batched, CSR and BSR), single
#     row (nrows == 1), uncoalesced (duplicate entries inside a row),
#     fully-dense CSR storage, BSR with blocks that do not divide the matrix,
#     and nan/inf/-0.0 values (all ignored by the accessor);
#   * negative: dense tensors, CSC tensors, BSC tensors, COO tensors and
#     non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a
# view of the input's own storage (there is nothing to broadcast against) and
# its result is an int64 metadata tensor (nothing to differentiate).

# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _random_compressed(batch, n_rows, n_cols, nnz, gen):
    """Deterministic valid compressed-row structure for a batch of matrices.

    Returns ``(crow, cols)`` where ``crow`` has shape ``batch + (n_rows + 1,)``
    (non-decreasing, ``crow[..., 0] == 0`` and ``crow[..., -1] == nnz``) and
    ``cols`` has shape ``batch + (nnz,)`` with column indices in ``[0, n_cols)``.
    Entries are sorted by (row, col) and drawn with replacement, so duplicate
    entries inside a row (uncoalesced storage) are allowed and legitimate.
    """
    entries = tuple(batch) + (nnz,)
    rows = torch.randint(0, n_rows, entries, dtype=torch.long, generator=gen)
    cols = torch.randint(0, n_cols, entries, dtype=torch.long, generator=gen)
    order = torch.argsort(rows * n_cols + cols, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)

    counts = torch.zeros(tuple(batch) + (n_rows,), dtype=torch.long)
    if nnz > 0:
        counts.scatter_add_(-1, rows, torch.ones(entries, dtype=torch.long))
    crow = torch.zeros(tuple(batch) + (n_rows + 1,), dtype=torch.long)
    crow[..., 1:] = counts.cumsum(-1)
    return crow, cols


def _make_values(dtype, shape, value_range):
    """Stored values from the shared value-range framework (tu.make_input).

    The returned col_indices view never depends on the values, so this one
    constructor covers every per-dtype range. Unsigned integer storage cannot
    represent the negative bound of the ``[-1, 0]`` / ``[min, 0]`` ranges;
    those ranges are snapped to the representable degenerate ``[0, 0]`` range
    (the metadata output is unaffected either way).
    """
    try:
        return tu.make_input(dtype, shape, list(value_range))
    except RuntimeError:
        return tu.make_input(dtype, shape, ["0", "0"])


def _build_csr(shape, nnz, dtype, value_range, seed=0):
    batch, n_rows, n_cols = shape[:-2], shape[-2], shape[-1]
    gen = torch.Generator("cpu").manual_seed(seed)
    crow, cols = _random_compressed(batch, n_rows, n_cols, nnz, gen)
    values = _make_values(dtype, tuple(batch) + (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _build_bsr(shape, nnz, blocks, dtype, value_range, seed=0):
    batch, n_rows, n_cols = shape[:-2], shape[-2], shape[-1]
    block_rows, block_cols = blocks
    # ceil keeps the compressed extents valid for blocks that do not divide the
    # matrix dims; torch.sparse_bsr_tensor infers the block size from the
    # trailing dims of the values tensor and pads the logical size internally.
    n_row_blocks = (n_rows + block_rows - 1) // block_rows
    n_col_blocks = (n_cols + block_cols - 1) // block_cols
    gen = torch.Generator("cpu").manual_seed(seed)
    crow, cols = _random_compressed(batch, n_row_blocks, n_col_blocks, nnz, gen)
    values = _make_values(
        dtype, tuple(batch) + (nnz, block_rows, block_cols), value_range
    )
    return torch.sparse_bsr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _build_input(layout, shape, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    if layout == "csr":
        return _build_csr(shape, nnz, dtype, value_range, seed)
    if layout in ("bsr", "bsr_batch"):
        return _build_bsr(shape, nnz, blocks, dtype, value_range, seed)
    raise ValueError(f"unknown layout {layout}")


def _probe_bsr(batched):
    shape = (2, 4, 6) if batched else (4, 6)
    try:
        inp = _build_bsr(shape, 4, (2, 2), torch.float32, ["-1", "1"])
        out = torch.ops.aten.col_indices(inp)
    except Exception:
        return False
    expected = tuple(shape[:-2]) + (4,)
    return out.dtype == torch.int64 and out.shape == expected


_BSR_SUPPORTED = _probe_bsr(False)
_BSR_BATCH_SUPPORTED = _probe_bsr(True)


# ---------------------------------------------------------------------------
# Layout cases by level
# ---------------------------------------------------------------------------

# (layout, shape, nnz, blocks): 2-D CSR (incl. single-row, square, full), 3-D /
# 4-D batched CSR, and 2-D BSR with varied block shapes.
_COL_CASES_CORE = [
    ("csr", (5, 4), 7, None),
    ("csr", (3, 8), 16, None),
    ("csr", (8, 3), 12, None),
    ("csr", (4, 4), 16, None),
    ("csr", (1, 6), 4, None),
    ("csr", (3, 5, 4), 7, None),
    ("csr", (2, 4, 6), 12, None),
    ("csr", (2, 3, 4, 5), 8, None),
    ("bsr", (4, 6), 4, (2, 2)),
    ("bsr", (8, 8), 6, (2, 2)),
]

# Higher-rank layouts for the "all" level (default, no --quick): multi-batch-dim
# batched CSR (ranks 5-7) and BSR with blocks that do not divide the matrix,
# plus batched BSR.
_COL_CASES_ALL = [
    ("csr", (12, 9, 3, 6), 9, None),
    ("csr", (3, 6, 4, 4, 6, 5), 11, None),
    ("csr", (7, 3, 12, 4, 2, 15), 10, None),
    ("csr", (3, 4, 2, 5, 3, 4, 2), 13, None),
    ("bsr", (10, 10), 6, (3, 4)),
    ("bsr_batch", (2, 4, 6), 4, (2, 2)),
    ("bsr_batch", (2, 8, 12), 6, (4, 4)),
]


def _supported(case):
    layout = case[0]
    if layout == "bsr":
        return _BSR_SUPPORTED
    if layout == "bsr_batch":
        return _BSR_BATCH_SUPPORTED
    return True


def _select(cases):
    return [case for case in cases if _supported(case)]


def _col_cases():
    """Layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return _select([("csr", (2, 19, 7), 8, None)])
    return _select(_COL_CASES_CORE + _COL_CASES_ALL)


def _col_value_range_cases():
    """Representative 2-D / batched CSR + BSR layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return _select([("csr", (2, 19, 7), 8, None)])
    return _select(
        [
            ("csr", (5, 4), 7, None),
            ("csr", (3, 5, 4), 7, None),
            ("bsr", (4, 6), 4, (2, 2)),
        ]
    )


# ---------------------------------------------------------------------------
# Storage dtypes
# ---------------------------------------------------------------------------

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse row-compressed runtime supports: every required
# spec dtype (int8/uint8/fp8/...) plus the wider float/int families and bool.
_COL_DTYPE_CANDIDATES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]


def _probe_dtype(dtype):
    """A dtype is supported when a tiny CSR tensor of that storage dtype can be
    built on the active device and col_indices returns the expected int64 view."""
    try:
        inp = _build_csr((3, 4), 5, dtype, ["-1", "1"])
        out = torch.ops.aten.col_indices(inp)
    except Exception:
        return False
    return out.dtype == torch.int64 and out.shape == (5,)


_COL_DTYPES = [dtype for dtype in _COL_DTYPE_CANDIDATES if _probe_dtype(dtype)]
if not _COL_DTYPES:  # pragma: no cover - sparse CSR unsupported on this backend
    _COL_DTYPES = [torch.float32]

_NAN_INF_DTYPES = [dtype for dtype in utils.ALL_FLOAT_DTYPES if dtype in _COL_DTYPES]


# ---------------------------------------------------------------------------
# Reference / candidate resolution and assertions
# ---------------------------------------------------------------------------


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
    # col_indices returns a view of the input's internal batch_dims + (nnz,)
    # int64 column index tensor. The entries are exact, and the schema
    # annotation Tensor(a) self -> Tensor(a) requires the result to alias the
    # input's col_indices storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert ref_out.shape == inp.col_indices().shape
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal col_indices tensor (both on the candidate and the reference).
    assert res_out.data_ptr() == inp.col_indices().data_ptr()
    assert ref_out.data_ptr() == ref_inp.col_indices().data_ptr()
    # Non-mutation: the accessor must leave the input's compressed structure
    # and stored values untouched. Values may legitimately hold nan/inf, so
    # compare float storage with equal_nan.
    utils.gems_assert_equal(inp.crow_indices(), ref_inp.crow_indices())
    utils.gems_assert_equal(inp.col_indices(), ref_inp.col_indices())
    if inp.dtype.is_floating_point or inp.dtype.is_complex:
        utils.gems_assert_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp.values(), ref_inp.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.col_indices
@pytest.mark.parametrize("case", _col_cases())
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype. The returned (batch_dims + (nnz,)) col_indices
    # view must match the reference exactly and alias the input's col_indices
    # storage.
    layout, shape, nnz, blocks = case
    inp = _build_input(layout, shape, nnz, blocks, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("case", _col_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the returned col_indices view never changes
    # because col_indices reads only layout metadata, not the values payload.
    layout, shape, nnz, blocks = case
    inp = _build_input(layout, shape, nnz, blocks, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_empty(dtype):
    # nnz == 0: cols and values are empty, but col_indices must still return a
    # (0,) int64 tensor (not a dense or wrongly-shaped tensor). The empty view
    # has a null data pointer on both sides, so the alias check degenerates to
    # 0 == 0.
    inp = _build_input("csr", (4, 5), 0, None, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_empty_batched(dtype):
    # nnz == 0 with batch dims: the returned col_indices preserves the batch
    # dims and has shape batch_dims + (0,).
    inp = _build_input("csr", (2, 4, 5), 0, None, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.skipif(not _BSR_SUPPORTED, reason="BSR col_indices unsupported")
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_empty_bsr(dtype):
    # nnz == 0 for BSR: col and values are empty, but col_indices must still
    # return a (0,) contiguous int64 view.
    inp = _build_input("bsr", (4, 6), 0, (2, 2), dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_single_row(dtype):
    # nrows == 1: crow has the degenerate shape (2,) with crow[0] == 0 and
    # crow[1] == nnz, and col_indices has shape (nnz,).
    inp = _build_input("csr", (1, 7), 5, None, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

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
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_full_storage(dtype):
    # Fully-dense CSR storage: every logical position is stored, so the crow
    # pointer array lists the cumulative row counts and col_indices enumerates
    # every column in row order.
    shape = (2, 3)
    crow = torch.tensor([0, 3, 6], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.skipif(not _BSR_SUPPORTED, reason="BSR col_indices unsupported")
@pytest.mark.parametrize("dtype", _COL_DTYPES)
def test_col_indices_bsr_ragged_blocks(dtype):
    # BSR whose blocks do not divide the matrix dims: the compressed extents
    # use ceil and torch pads the logical size internally. col_indices returns
    # the stored block-column indices, one per stored block.
    inp = _build_input("bsr", (10, 10), 6, (3, 4), dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_col_indices_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/+-0.0 are ordinary stored values: col_indices must still
    # return exactly the stored col_indices tensor, unchanged, for every one of
    # them.
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device=flag_gems.device)
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
def test_col_indices_dense_raises():
    # col_indices dispatches only on sparse row-compressed tensors; dense
    # tensors have no implementation and raise. The candidate must fail too
    # rather than silently return a bogus col_indices tensor.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices
def test_col_indices_csc_raises():
    # SparseCsc is a distinct compressed layout from SparseCsr; col_indices
    # requires a row-compressed layout and raises on CSC. The candidate must
    # reject it too.
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values.to(flag_gems.device), (4, 2)
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices
def test_col_indices_bsc_raises():
    # SparseBsc is a column-compressed blocked layout, not row compressed;
    # col_indices raises on it and so must the candidate.
    ccol_indices = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1], dtype=torch.long, device=flag_gems.device)
    values = torch.randn(2, 2, 2, dtype=torch.float32, device=flag_gems.device)
    inp = torch.sparse_bsc_tensor(
        ccol_indices, row_indices, values, (4, 4), device=flag_gems.device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices
def test_col_indices_coo_raises():
    # Sparse (COO) is a distinct backend key from SparseCsr; col_indices has no
    # Sparse implementation and raises. The candidate must reject it too.
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices
def test_col_indices_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
