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
from . import test_utils as tu  # noqa: E402

# aten::ccol_indices(Tensor(a) self) -> Tensor(a) returns the compressed column
# index tensor of a sparse CSC tensor: shape batch_dims + (ncols + 1,) (the
# unbatched 2-D case has batch_dims == ()) with dtype int64. The result is an
# alias of the input's internal ccol storage and never depends on the stored
# values, so every workload below feeds a sparse CSC tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (shape, nnz) layouts from the quick/core/all levels, ranks
#     2-7 (2-D all-sparse, 3-D/4-D batched, and higher-rank multi-batch-dims),
#     with varying nnz so the (batch_dims + (ncols + 1,)) shape of the result
#     is exercised;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the returned ccol is identical for all of them);
#   * edge cases: empty (nnz == 0, unbatched and batched), single column
#     (ncols == 1), uncoalesced (duplicate row entries inside a column),
#     fully-dense CSC storage, and nan/inf/-0.0 values (all ignored by the
#     accessor);
#   * negative: dense tensors, CSR tensors, COO tensors and non-tensor inputs
#     are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a view
# of the input's own storage (there is nothing to broadcast against) and its
# result is an int64 metadata tensor (nothing to differentiate).

# (shape, nnz) layouts covering 2-D all-sparse, 3-D batched, and 4-D
# multi-batch-dims.
_CSC_CASES_CORE = [
    ((5, 4), 7),
    ((3, 8), 16),
    ((8, 3), 12),
    ((4, 4), 16),
    ((1, 6), 4),
    ((3, 5, 4), 7),
    ((2, 4, 6), 12),
    ((2, 3, 4, 5), 8),
]

# Higher-rank layouts for the "all"/"extended" TEST_LEVEL: 4-D and batched
# ranks up to 7-D.
_CSC_CASES_ALL = [
    ((12, 9, 3, 6), 9),
    ((3, 6, 4, 4, 6, 5), 11),
    ((7, 3, 12, 4, 2, 15), 10),
    ((3, 4, 2, 5, 3, 4, 2), 13),
]


def _csc_cases():
    """(shape, nnz) layouts selected by the TEST_LEVEL env var."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL in ("all", "extended"):
        return _CSC_CASES_CORE + _CSC_CASES_ALL
    return _CSC_CASES_CORE


def _csc_value_range_cases():
    """Representative 2-D + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL in ("all", "extended"):
        return [((5, 4), 7), ((3, 5, 4), 7), ((3, 6, 4, 4, 6, 5), 11)]
    return [((5, 4), 7), ((3, 5, 4), 7)]


# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSC runtime supports: every float, int, and bool
# family.
_CSC_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side (row, col) generation; the values tensor comes
    # from the shared value-range helper (tu.make_input) and the sparse tensor
    # is created on the test device. Duplicate entries are allowed and merely
    # leave the tensor uncoalesced (covered explicitly below). The ccol pointer
    # array is built with a (vectorized, per-batch) column-wise bincount, so it
    # is always a valid CSC structure.
    gen = torch.Generator("cpu").manual_seed(seed)
    nrows, ncols = shape[-2], shape[-1]
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, generator=gen)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, generator=gen)
    order = torch.argsort(cols * nrows + rows, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    batch_numel = 1
    for dim in batch:
        batch_numel *= dim
    offset = (torch.arange(batch_numel, dtype=torch.long) * ncols).view(batch_numel, 1)
    flat = (cols.reshape(batch_numel, nnz) + offset).reshape(-1)
    counts = torch.bincount(flat, minlength=batch_numel * ncols).view(
        batch_numel, ncols
    )
    ccol = torch.zeros(batch_numel, ncols + 1, dtype=torch.long)
    ccol[:, 1:] = torch.cumsum(counts, -1)
    ccol = ccol.view(batch + (ncols + 1,))
    values = tu.make_input(dtype, entries_shape, value_range)
    return torch.sparse_csc_tensor(
        ccol.to(flag_gems.device),
        rows.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.ccol_indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems.ccol_indices callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices", getattr(flag_gems, "ccol_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # ccol_indices returns a fresh view of the input's internal
    # batch_dims + (ncols + 1,) int64 compressed column index tensor. The
    # entries are exact, and the schema annotation Tensor(a) self -> Tensor(a)
    # requires the result to alias the input's ccol storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert ref_out.shape == inp.shape[:-2] + (inp.shape[-1] + 1,)
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal ccol tensor (both on the candidate and the reference).
    assert res_out.data_ptr() == torch.ops.aten.ccol_indices(inp).data_ptr()
    assert ref_out.data_ptr() == torch.ops.aten.ccol_indices(ref_inp).data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set), so its ccol, row indices and
    # values still match the (untouched) input storage after the calls. Values
    # may legitimately hold nan/inf, so compare them with equal_nan for float
    # storage.
    utils.gems_assert_equal(inp.ccol_indices(), ref_inp.ccol_indices())
    utils.gems_assert_equal(inp.row_indices(), ref_inp.row_indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp.values(), ref_inp.values())


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _csc_cases())
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The returned (batch_dims + (ncols + 1,)) ccol view must match the
    # reference exactly and alias the input's ccol storage.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _csc_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the returned ccol view never changes because
    # ccol_indices reads only layout metadata, not the values payload.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_empty(dtype):
    # nnz == 0: rows and values are empty, but ccol_indices must still return a
    # (ncols + 1,) int64 tensor (not a dense or wrongly-shaped tensor).
    shape = (4, 5)
    ccol = torch.zeros(6, dtype=torch.long, device=flag_gems.device)
    rows = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_empty_batched(dtype):
    # nnz == 0 with batch dims: the returned ccol preserves the batch_dims and
    # has shape batch_dims + (ncols + 1,).
    shape = (2, 4, 5)
    ccol = torch.zeros(2, 6, dtype=torch.long, device=flag_gems.device)
    rows = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(2, 0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_single_column(dtype):
    # ncols == 1: the returned ccol has the degenerate shape (2,) with
    # ccol[0] == 0 and ccol[1] == nnz.
    shape, nnz = (7, 1), 5
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (rows[0] == rows[1] in column 0), which
    # leaves the tensor uncoalesced; ccol_indices must still return exactly the
    # stored ccol tensor (never a coalesced/sorted copy). Column 0 holds 3
    # entries for rows [0, 0, 2], so a coalescing implementation would visibly
    # change the stored structure.
    shape = (4, 3)
    ccol = torch.tensor([0, 3, 3, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert rows[0].item() == rows[1].item()
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(ccol, rows, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_ccol_indices_full_storage(dtype):
    # Fully-dense CSC storage: every logical position is stored, so the ccol
    # pointer array lists the cumulative counts of every column.
    shape = (2, 3)
    ccol = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=flag_gems.device)
    rows = torch.arange(2).repeat(3).to(flag_gems.device)  # [0, 1, 0, 1, 0, 1]
    values = tu.make_input(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(ccol, rows, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_ccol_indices_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: ccol_indices must still
    # return exactly the stored ccol tensor, unchanged, for every one of them.
    shape = (3, 4)
    ccol = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
def test_ccol_indices_dense_raises():
    # ccol_indices dispatches only on the SparseCsr (CSC) backend key; dense
    # tensors have no implementation and raise. The candidate must fail too
    # rather than silently return a bogus ccol tensor.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_csr_raises():
    # SparseCsr is a distinct compressed layout from SparseCsc; ccol_indices
    # has no SparseCsr implementation and raises. The candidate must reject it
    # too.
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values.to(flag_gems.device), (2, 4)
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_coo_raises():
    # Sparse (COO) is a distinct backend key from SparseCsr (CSC); ccol_indices
    # has no Sparse implementation and raises. The candidate must reject it too.
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices
def test_ccol_indices_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten.ccol_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
