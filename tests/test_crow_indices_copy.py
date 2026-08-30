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

# aten::crow_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (nrows + 1,) int64 compressed row index tensor of a sparse CSR
# tensor as a fresh, contiguous, independent copy (the view_copy counterpart of
# aten::crow_indices, whose native body is
# `crow_indices(self).clone(contiguous)`). Unlike aten::crow_indices the result
# does NOT alias the input's internal crow storage and never depends on the
# stored values, so every workload below feeds a sparse CSR tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (shape, nnz) layouts from the quick/core/all levels, ranks
#     2-7 (2-D all-sparse, 3-D/4-D batched, and higher-rank multi-batch-dims),
#     with varying nnz so the (batch_dims + (nrows + 1,)) shape of the result
#     is exercised;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the returned crow copy is identical for all of
#     them);
#   * edge cases: empty (nnz == 0, unbatched and batched), single row
#     (nrows == 1), uncoalesced (duplicate column entries inside a row),
#     fully-dense CSR storage, and nan/inf/-0.0 values (all ignored by the
#     accessor);
#   * .out overload: the (self, *, out) variant must write the same fresh copy
#     into out and return out itself (empty and non-empty layouts);
#   * negative: dense tensors, CSC tensors, COO tensors and non-tensor inputs
#     are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns an
# independent copy of a metadata tensor derived from the input's own storage
# (there is nothing to broadcast against) and its result is an int64 metadata
# tensor (nothing to differentiate).

# (shape, nnz) layouts covering 2-D all-sparse, 3-D batched, and 4-D
# multi-batch-dims.
_CSR_COPY_CASES_CORE = [
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
_CSR_COPY_CASES_ALL = [
    ((12, 9, 3, 6), 9),
    ((3, 6, 4, 4, 6, 5), 11),
    ((7, 3, 12, 4, 2, 15), 10),
    ((3, 4, 2, 5, 3, 4, 2), 13),
]


def _csr_copy_cases():
    """(shape, nnz) layouts selected by the TEST_LEVEL env var."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL in ("all", "extended"):
        return _CSR_COPY_CASES_CORE + _CSR_COPY_CASES_ALL
    return _CSR_COPY_CASES_CORE


def _csr_copy_value_range_cases():
    """Representative 2-D + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 8)]
    if tu.LEVEL in ("all", "extended"):
        return [((5, 4), 7), ((3, 5, 4), 7), ((3, 6, 4, 4, 6, 5), 11)]
    return [((5, 4), 7), ((3, 5, 4), 7)]


# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSR runtime supports: every float, int, and bool
# family.
_CSR_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side (row, col) generation; the values tensor comes
    # from the shared value-range helper (tu.make_input) and the sparse tensor
    # is created on the test device. Duplicate entries are allowed and merely
    # leave the tensor uncoalesced (covered explicitly below). The crow pointer
    # array is built with a (vectorized, per-batch) row-wise bincount, so it is
    # always a valid CSR structure.
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
    values = tu.make_input(dtype, entries_shape, value_range)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _reference_crow_indices_copy(inp):
    # Prefer the literal ATen op as the reference. Some PyTorch builds register
    # crow_indices_copy as CompositeExplicitAutogradNonFunctional, whose
    # dispatch-key set excludes the SparseCsr functionality key, so calling
    # torch.ops.aten.crow_indices_copy directly on a sparse CSR tensor raises
    # NotImplementedError. In that case fall back to the operator's exact
    # native body -- crow_indices(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse CSR tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten.crow_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.crow_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _reference_crow_indices_copy_out(inp, out):
    # Same strategy as _reference_crow_indices_copy for the .out overload:
    # compute the materialized copy and write it into out (the .out contract
    # returns out itself).
    try:
        return torch.ops.aten.crow_indices_copy.out(inp, out=out)
    except NotImplementedError:
        computed = torch.ops.aten.crow_indices(inp).clone(
            memory_format=torch.contiguous_format
        )
        torch.ops.aten.copy_(out, computed)
        return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.crow_indices_copy is registered;
    # resolution order is: (1) override, (2) the direct
    # flag_gems.crow_indices_copy callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "crow_indices_copy", getattr(flag_gems, "crow_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "crow_indices_copy.out", getattr(flag_gems, "crow_indices_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp):
    # crow_indices_copy returns a fresh contiguous batch_dims + (nrows + 1,)
    # int64 tensor holding the input's raw crow indices. The entries are exact,
    # and copy semantics require the result to NOT alias the input's crow
    # storage (unlike aten::crow_indices) and to not mutate the input.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert ref.shape == inp.shape[:-2] + (inp.shape[-2] + 1,)
    assert res.shape == ref.shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's crow indices.
    # Zero-size results (nrows + 1 == 0) may share the null pointer, so only
    # assert on non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != torch.ops.aten.crow_indices(inp).data_ptr()
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


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _csr_copy_cases())
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The returned (batch_dims + (nrows + 1,)) crow copy must match the
    # reference exactly, be contiguous, and hold fresh storage.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case", _csr_copy_cases())
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_out(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    out_shape = shape[:-2] + (shape[-2] + 1,)
    out = torch.empty(out_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(out_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_crow_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _csr_copy_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the returned crow copy never changes because
    # crow_indices_copy reads only layout metadata, not the values payload.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_empty(dtype):
    # nnz == 0: cols and values are empty, but crow_indices_copy must still
    # return a (nrows + 1,) contiguous int64 tensor (not a dense or
    # wrongly-shaped tensor).
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_out_empty(dtype):
    # nnz == 0: the .out variant must still write a (nrows + 1,) int64 tensor
    # into out and return out itself.
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    out_shape = shape[:-2] + (shape[-2] + 1,)
    out = torch.empty(out_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(out_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_crow_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_empty_batched(dtype):
    # nnz == 0 with batch dims: the returned crow preserves the batch_dims and
    # has shape batch_dims + (nrows + 1,).
    shape, nnz = (2, 4, 5), 0
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_single_row(dtype):
    # nrows == 1: the returned crow has the degenerate shape (2,) with
    # crow[0] == 0 and crow[1] == nnz.
    shape, nnz = (1, 7), 5
    inp = _make_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (cols[0] == cols[1] in row 0), which
    # leaves the tensor uncoalesced; crow_indices_copy must still return exactly
    # the stored crow tensor as an independent copy (never a coalesced/sorted
    # structure and never an alias). Row 0 holds 3 entries for columns
    # [0, 0, 2], so a coalescing implementation would visibly change the stored
    # structure.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CSR_COPY_DTYPES)
def test_crow_indices_copy_full_storage(dtype):
    # Fully-dense CSR storage: every logical position is stored, so the crow
    # pointer array lists the cumulative counts of every row.
    shape = (2, 3)
    crow = torch.tensor([0, 3, 6], dtype=torch.long, device=flag_gems.device)
    cols = torch.arange(3).repeat(2).to(flag_gems.device)  # [0, 1, 2, 0, 1, 2]
    values = tu.make_input(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_crow_indices_copy_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: crow_indices_copy must
    # still return exactly the stored crow tensor, unchanged, for every one of
    # them.
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

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_dense_raises():
    # crow_indices_copy dispatches only on the SparseCsr (CSR) backend key;
    # dense tensors have no implementation and raise. The candidate must fail
    # too rather than silently return a bogus crow tensor.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices_copy(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_csc_raises():
    # SparseCsr is a distinct compressed layout from SparseCsc; crow_indices_copy
    # has no SparseCsc implementation and raises. The candidate must reject it
    # too.
    ccol = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor(
        [0, 1, 0, 1, 0, 1], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_input(torch.float32, (6,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol, row_indices, values.to(flag_gems.device), (2, 3)
    )
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices_copy(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_coo_raises():
    # Sparse (COO) is a distinct backend key from SparseCsr (CSR);
    # crow_indices_copy has no Sparse implementation and raises. The candidate
    # must reject it too.
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.aten.crow_indices_copy(utils.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        _resolve_gems_op()(inp)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.crow_indices_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
