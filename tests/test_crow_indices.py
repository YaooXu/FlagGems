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

import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::crow_indices(Tensor(a) self) -> Tensor(a) returns the compressed row
# index tensor of a sparse CSR tensor: shape batch_dims + (nrows + 1,) with
# dtype int64. The result is an alias of the input's internal crow storage
# (Tensor(a) -> Tensor(a)) and never depends on the stored values, so every
# workload below feeds a sparse CSR tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * dtype coverage: the 9 required spec dtypes (int8, uint8, float8_e4m3fn,
#     float8_e5m2, float32, bfloat16, float16, int32, int64) plus float64,
#     int16 and bool. Every candidate is probed with a real CSR tensor before
#     being parametrized (some backends cannot build fp8/sparse storage), and
#     the operator reads only the crow metadata regardless of storage dtype;
#   * shape levels: crow_indices only accepts rank >= 2 CSR layouts, so the
#     spec's 0-dim/1-dim levels are represented by their nearest CSR-valid
#     analogues -- ((1, 1)) for the scalar/single-element boundary and ((1, 6))
#     for the single-row boundary -- together with the 2-D (256, 256) /
#     (1024, 1024), 3-D (20, 320, 15), 4-D (16, 128, 64, 60) and 5-D
#     (16, 7, 57, 32, 29) regular levels and higher-rank multi-batch-dims
#     layouts, all from the quick/all levels;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the returned crow is identical for all of them);
#   * edge cases: empty (nnz == 0, unbatched and batched), single row
#     (nrows == 1), uncoalesced (duplicate column entries inside a row),
#     fully-dense CSR storage, and nan/inf/-0.0 values (all ignored by the
#     accessor);
#   * negative cases: dense tensors, CSC tensors, COO tensors and non-tensor
#     inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a view
# of the input's own storage (there is nothing to broadcast against) and its
# result is an int64 metadata tensor (nothing to differentiate).

# (shape, nnz) layouts covering the CSR-valid analogues of the seven spec shape
# levels (single element, single row, 2-D regular, 3-D, 4-D, 5-D) plus a large
# 2-D layout.
_CSR_CASES_CORE = [
    ((1, 1), 1),
    ((1, 6), 4),
    ((5, 4), 7),
    ((256, 256), 512),
    ((1024, 1024), 4096),
    ((3, 5, 4), 7),
    ((20, 320, 15), 100),
    ((16, 128, 64, 60), 50),
    ((16, 7, 57, 32, 29), 5),
]

# Higher-rank / batched layouts for the "all" level (no --quick): 2-D all-sparse
# variations, 3-D/4-D batched and ranks up to 7-D.
_CSR_CASES_ALL = [
    ((3, 8), 16),
    ((8, 3), 12),
    ((4, 4), 16),
    ((2, 4, 6), 12),
    ((2, 3, 4, 5), 8),
    ((12, 9, 3, 6), 9),
    ((3, 6, 4, 4, 6, 5), 11),
    ((7, 3, 12, 4, 2, 15), 10),
    ((3, 4, 2, 5, 3, 4, 2), 13),
]


def _csr_cases():
    """(shape, nnz) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL == "all":
        return _CSR_CASES_CORE + _CSR_CASES_ALL


def _csr_value_range_cases():
    """Representative 2-D + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL == "all":
        return [((5, 4), 7), ((3, 5, 4), 7), ((3, 6, 4, 4, 6, 5), 11)]


# Every spec dtype (int8/uint8/fp8 are hard requirements) plus the float/int/bool
# families from accuracy_utils. The actual parametrization is the probed subset:
# the operator accepts any storage dtype the CSR runtime can hold.
_CSR_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + utils.BOOL_TYPES
        + [
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
    )
)


def _probe_csr_dtypes(candidates):
    """Probe which storage dtypes can build a CSR tensor and run crow_indices."""
    supported = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dtype in candidates:
            try:
                crow = torch.tensor([0, 1], dtype=torch.long, device=flag_gems.device)
                cols = torch.tensor([0], dtype=torch.long, device=flag_gems.device)
                values = torch.ones(1, dtype=dtype, device=flag_gems.device)
                inp = torch.sparse_csr_tensor(crow, cols, values, (1, 1))
                torch.ops.aten.crow_indices(inp)
            except Exception:
                continue
            supported.append(dtype)
    return supported


# Fallback keeps the full candidate list rather than a float32-only one, so a
# failed/absent probe never silently drops the spec-required int8/uint8/fp8
# dtypes.
_CSR_DTYPES = _probe_csr_dtypes(_CSR_DTYPE_CANDIDATES) or list(_CSR_DTYPE_CANDIDATES)


def _make_values(dtype, shape, value_range):
    """Value-range helper with an unsigned-dtype fallback.

    tu.make_input is used whenever possible. Unsigned dtypes (e.g. uint8) have
    no negative values, so the [-1, 0] range clamps to a degenerate 0/0 range
    and torch.testing.make_tensor rejects it; clamp both bounds into the dtype
    bounds and fill the constant instead (still inside the requested range).
    """
    try:
        return tu.make_input(dtype, shape, value_range)
    except RuntimeError:
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        lo_bound, hi_bound = tu.dtype_bounds(dtype)
        if not (dtype.is_floating_point or dtype.is_complex):
            low, high = int(low), int(high)
            lo_bound, hi_bound = int(lo_bound), int(hi_bound)
        low = min(max(low, lo_bound), hi_bound)
        high = min(max(high, lo_bound), hi_bound)
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )


def _make_input(shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side (row, col) generation; the values tensor comes
    # from the shared value-range helper and the sparse tensor is created on the
    # test device. Duplicate entries are allowed and merely leave the tensor
    # uncoalesced (covered explicitly below). The crow pointer array is built
    # with a (vectorized, per-batch) row-wise bincount, so it is always a valid
    # CSR structure.
    gen = torch.Generator("cpu").manual_seed(seed)
    nrows, ncols = shape[-2], shape[-1]
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, generator=gen)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, generator=gen)
    order = torch.argsort(rows * ncols + cols, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    batch_numel = 1
    for dim in batch:
        batch_numel *= dim
    offset = (torch.arange(batch_numel, dtype=torch.long) * nrows).view(batch_numel, 1)
    flat = (rows.reshape(batch_numel, nnz) + offset).reshape(-1)
    counts = torch.bincount(flat, minlength=batch_numel * nrows).view(
        batch_numel, nrows
    )
    crow = torch.zeros(batch_numel, nrows + 1, dtype=torch.long)
    crow[:, 1:] = torch.cumsum(counts, -1)
    crow = crow.view(batch + (nrows + 1,))
    values = _make_values(dtype, entries_shape, value_range)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.crow_indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems.crow_indices callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "crow_indices", getattr(flag_gems, "crow_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # crow_indices returns a view of the input's internal
    # batch_dims + (nrows + 1,) int64 compressed row index tensor. The entries
    # are exact, and the schema annotation Tensor(a) self -> Tensor(a) requires
    # the result to alias the input's crow storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert ref_out.shape == inp.shape[:-2] + (inp.shape[-2] + 1,)
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal crow tensor (both on the candidate and the reference).
    assert res_out.data_ptr() == torch.ops.aten.crow_indices(inp).data_ptr()
    assert ref_out.data_ptr() == torch.ops.aten.crow_indices(ref_inp).data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set), so its crow, col indices and
    # values still match the (untouched) input storage after the calls. Values
    # may legitimately hold nan/inf, so compare them with equal_nan for float
    # storage.
    utils.gems_assert_equal(inp.crow_indices(), ref_inp.crow_indices())
    utils.gems_assert_equal(inp.col_indices(), ref_inp.col_indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp.values(), ref_inp.values())


@pytest.mark.crow_indices
@pytest.mark.parametrize("case", _csr_cases())
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every probed storage dtype (bool/int snap the range to the
    # representable set). The returned (batch_dims + (nrows + 1,)) crow view
    # must match the reference exactly and alias the input's crow storage.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("case", _csr_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the returned crow view never changes because
    # crow_indices reads only layout metadata, not the values payload.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_empty(dtype):
    # nnz == 0: cols and values are empty, but crow_indices must still return a
    # (nrows + 1,) int64 tensor (not a dense or wrongly-shaped tensor).
    shape = (4, 5)
    crow = torch.zeros(5, dtype=torch.long, device=flag_gems.device)
    cols = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_empty_batched(dtype):
    # nnz == 0 with batch dims: the returned crow preserves the batch_dims and
    # has shape batch_dims + (nrows + 1,).
    shape = (2, 4, 5)
    crow = torch.zeros(2, 5, dtype=torch.long, device=flag_gems.device)
    cols = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(2, 0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_single_row(dtype):
    # nrows == 1: the returned crow has the degenerate shape (2,) with
    # crow[0] == 0 and crow[1] == nnz.
    shape, nnz = (1, 7), 5
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (cols[0] == cols[1] in row 0), which
    # leaves the tensor uncoalesced; crow_indices must still return exactly the
    # stored crow tensor (never a coalesced/sorted copy). Row 0 holds 3 entries
    # for columns [0, 0, 2], so a coalescing implementation would visibly
    # change the stored structure.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_crow_indices_full_storage(dtype):
    # Fully-dense CSR storage: every logical position is stored, so the crow
    # pointer array lists the cumulative counts of every row.
    shape = (2, 3)
    crow = torch.tensor([0, 3, 6], dtype=torch.long, device=flag_gems.device)
    cols = torch.arange(3).repeat(2).to(flag_gems.device)  # [0, 1, 2, 0, 1, 2]
    values = _make_values(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_crow_indices_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: crow_indices must still
    # return exactly the stored crow tensor, unchanged, for every one of them.
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices
def test_crow_indices_dense_raises():
    # crow_indices dispatches only on the SparseCsr (CSR) backend key; dense
    # tensors have no implementation and raise. The candidate must fail too
    # rather than silently return a bogus crow tensor.
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices
def test_crow_indices_csc_raises():
    # SparseCsr is a distinct compressed layout from SparseCsc; crow_indices
    # has no SparseCsc implementation and raises. The candidate must reject it
    # too.
    ccol = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor(
        [0, 1, 0, 1, 0, 1], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values(torch.float32, (6,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol, row_indices, values.to(flag_gems.device), (2, 3)
    )
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices
def test_crow_indices_coo_raises():
    # Sparse (COO) is a distinct backend key from SparseCsr (CSR); crow_indices
    # has no Sparse implementation and raises. The candidate must reject it too.
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices
def test_crow_indices_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
