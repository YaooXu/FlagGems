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

import math
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

# aten::col_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (nnz,) int64 column index tensor of a sparse row-compressed
# tensor (CSR or BSR) as a fresh, contiguous, independent copy (the view_copy
# counterpart of aten::col_indices, whose native body is `col_indices(self).
# clone(contiguous)`). Every workload feeds a sparse CSR or BSR tensor and
# checks copy semantics: the result must equal the raw col_indices array, must
# NOT alias the input's internal col_indices storage, and must not mutate the
# input. The returned tensor never depends on the stored values, so every
# supported storage dtype (all float/int families plus bool) is exercised
# through the same layouts.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (layout, size, nnz, blocks) cases from the quick/all
#     --quicks — 2-D CSR (incl. single-row, single-column, square, full
#     and nnz == 0), batched CSR (multi-batch-dims), 2-D BSR with varied block
#     shapes (incl. blocks that do not divide the matrix), and batched BSR,
#     ranks 2-5;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme
#     and degenerate value ranges (the returned col_indices copy is identical
#     for all of them);
#   * edge cases: empty (nnz == 0, CSR and BSR), uncoalesced (duplicate column
#     entries inside a row), and nan/inf/-0.0 values (all ignored by the
#     accessor);
#   * negative: dense tensors, CSC tensors, COO tensors, non-tensor inputs and
#     a wrong-dtype ``out`` tensor are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a
# fresh int64 metadata tensor (nothing to broadcast against or differentiate).

# (layout, size, nnz, blocks) core cases: 2-D CSR (incl. single-row/single-
# column/square/nnz==0), batched CSR, 2-D BSR with varied block shapes, batched
# BSR, and an empty BSR.
_COLS_CORE = [
    ("csr", (5, 4), 6, None),
    ("csr", (4, 1), 3, None),
    ("csr", (1, 5), 2, None),
    ("csr", (8, 8), 16, None),
    ("csr", (16, 32), 40, None),
    ("csr", (32, 16), 80, None),
    ("csr", (3, 3), 9, None),
    ("csr", (3, 4), 0, None),
    ("csr_batch", (2, 6, 8), 12, None),
    ("bsr", (4, 6), 4, (2, 2)),
    ("bsr", (8, 8), 8, (2, 2)),
    ("bsr", (6, 6), 6, (3, 2)),
    ("bsr_batch", (2, 4, 6), 6, (2, 2)),
]

# Higher-rank / wider layouts for the "all" level (default, no --quick): a multi-
# batch-dim batched CSR, a BSR whose blocks do not divide the matrix, and a
# batched BSR with a bigger block.
_COLS_ALL = [
    ("csr_batch", (7, 3, 12, 4, 5), 48, None),
    ("bsr", (10, 10), 12, (3, 4)),
    ("bsr_batch", (2, 8, 12), 12, (4, 4)),
]


def _cols_cases():
    """(layout, size, nnz, blocks) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [("csr_batch", (2, 19, 7), 20, None)]
    if tu.LEVEL == "all":
        return _COLS_CORE + _COLS_ALL


def _cols_value_range_cases():
    """Representative row-compressed + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [("csr", (5, 4), 6, None)]
    if tu.LEVEL == "all":
        return [
            ("csr", (5, 4), 6, None),
            ("csr_batch", (2, 6, 8), 12, None),
            ("bsr", (4, 6), 4, (2, 2)),
        ]
    return [("csr", (5, 4), 6, None), ("bsr", (4, 6), 4, (2, 2))]


# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSR/BSR runtime supports: every float, int, and
# bool family.
_COLS_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage uses float + int storage dtypes (bool ignores the range
# in tu.make_input and adds nothing beyond the copy-semantics tests above).
_VALUE_RANGE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# nan / +-inf stored values for the metadata-accessor test.
_NAN_INF_DTYPES = utils.ALL_FLOAT_DTYPES


def _random_crow(n_compressed, nnz, gen):
    # A valid compressed-row array: length n_compressed + 1, non-decreasing,
    # crow[0] == 0 and crow[-1] == nnz. Repeated split points leave empty rows,
    # which is valid for the compressed format. With n_compressed == 1 the
    # array is the degenerate [0, nnz].
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
    )


def _make_values(dtype, values_shape, value_range, gen):
    # Stored values come from the value-range framework (tu.make_input) because
    # the returned col_indices copy never depends on them; every per-dtype range
    # can therefore be exercised through this one constructor. bool ignores the
    # range (deterministic random 0/1 keeps the seed-based construction
    # reproducible).
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen).to(
            flag_gems.device
        )
    # tu.make_input already creates the tensor on flag_gems.device; the .to()
    # it to flag_gems.device so it always matches the index tensors.
    return tu.make_input(dtype, values_shape, list(value_range)).to(flag_gems.device)


def _make_csr(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    crow = _random_crow(n_rows, nnz, gen)
    col = torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)


def _make_csr_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    n_batch = 1
    for dim in batch_dims:
        n_batch *= dim
    crow = torch.stack([_random_crow(n_rows, nnz, gen) for _ in range(n_batch)])
    col = torch.randint(0, n_cols, (n_batch, nnz), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (n_batch, nnz), value_range, gen)
    return torch.sparse_csr_tensor(
        crow.view(batch_dims + (n_rows + 1,)),
        col.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz,)),
        size=size,
        device=device,
    )


def _make_bsr(size, nnz, blocks, dtype, gen, device, value_range):
    n_rows, n_cols = size
    block_rows, block_cols = blocks
    # ceil keeps the compressed extents valid for blocks that do not divide
    # the matrix dims; torch.sparse_bsr_tensor infers the block size from the
    # trailing dims of the values tensor and pads the logical size internally.
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    crow = _random_crow(n_row_blocks, nnz, gen)
    col = torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range, gen)
    return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)


def _make_bsr_batch(size, nnz, blocks, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    block_rows, block_cols = blocks
    n_batch = 1
    for dim in batch_dims:
        n_batch *= dim
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    crow = torch.stack([_random_crow(n_row_blocks, nnz, gen) for _ in range(n_batch)])
    col = torch.randint(
        0, n_col_blocks, (n_batch, nnz), dtype=torch.long, generator=gen
    )
    values = _make_values(
        dtype, (n_batch, nnz, block_rows, block_cols), value_range, gen
    )
    return torch.sparse_bsr_tensor(
        crow.view(batch_dims + (n_row_blocks + 1,)),
        col.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz, block_rows, block_cols)),
        size=size,
        device=device,
    )


def _make_input(layout, size, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    # Deterministic CPU-side compressed/column index generation; the values
    # tensor comes from the shared value-range helper (tu.make_input) and the
    # sparse tensor is created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csr":
        return _make_csr(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "csr_batch":
        return _make_csr_batch(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "bsr":
        return _make_bsr(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)
    return _make_bsr_batch(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)


def _expected_col_shape(case):
    # col_indices_copy returns batch_dims + (nnz,) int64 entries for both CSR
    # and BSR layouts (for BSR the entries are the block column indices, still
    # one per stored block).
    layout, size, nnz, blocks = case
    del blocks
    if layout == "csr":
        return (nnz,)
    if layout == "csr_batch":
        return size[:-2] + (nnz,)
    if layout == "bsr":
        return (nnz,)
    if layout == "bsr_batch":
        return size[:-2] + (nnz,)
    raise ValueError(f"unknown layout {layout}")


def _reference_col_indices_copy(inp):
    # Prefer the literal ATen op as the reference. The installed PyTorch build
    # registers col_indices_copy as CompositeExplicitAutogradNonFunctional,
    # whose dispatch-key set may exclude the SparseCsr functionality key, so
    # calling torch.ops.aten.col_indices_copy directly on a sparse tensor can
    # raise NotImplementedError. In that case fall back to the operator's exact
    # native body -- col_indices(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse row-compressed tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten.col_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.col_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _reference_col_indices_copy_out(inp, out):
    # Same strategy as _reference_col_indices_copy for the .out overload. The
    # .out contract materializes int64 entries into out and returns out itself;
    # ATen enforces the int64 out dtype, so the manual native-body fallback
    # (compute the materialized copy and copy it into out) enforces the same
    # contract up front to stay consistent across builds where the .out op is
    # unreachable on sparse tensors.
    if out.dtype != torch.int64:
        raise RuntimeError(
            "Expected out tensor to have dtype long int, "
            f"but got {out.dtype} instead"
        )
    try:
        return torch.ops.aten.col_indices_copy.out(inp, out=out)
    except NotImplementedError:
        computed = _reference_col_indices_copy(inp)
        torch.ops.aten.copy_(out, computed)
        return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.col_indices_copy is registered;
    # resolution order is: (1) override, (2) the direct
    # flag_gems.col_indices_copy callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "col_indices_copy", getattr(flag_gems, "col_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "col_indices_copy.out", getattr(flag_gems, "col_indices_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_shape):
    # col_indices_copy returns a fresh contiguous int64 tensor holding the
    # input's raw column index array (nnz entries, or batch_dims + nnz for
    # batched layouts). The result must not alias the input's internal
    # col_indices storage and the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == expected_shape
    assert ref.shape == expected_shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's col_indices.
    # For nnz == 0 the result and the input's internal col_indices storage are
    # both empty (data_ptr() == 0), so the pointer check only applies to
    # non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != inp.col_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set). equal_nan=True keeps the
    # non-mutation check valid for inputs whose stored values contain nan /
    # +-inf (test_col_indices_copy_nan_inf_values): a mutated tensor still
    # differs from the snapshot on the finite entries.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _cols_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _cols_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    expected_shape = _expected_col_shape(case)
    out = torch.empty(expected_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(expected_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, expected_shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _cols_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_value_ranges(case, value_range, dtype):
    # Value-range coverage from the regular-operator spec: the metadata output
    # (the fresh col_indices int64 copy) is independent of the stored values,
    # so every per-dtype range -- including the extreme [min, 0] and [0, max]
    # magnitudes -- must be accepted and must not perturb the returned column
    # index array.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _cols_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_out_value_ranges(case, value_range, dtype):
    # Same sweep through the .out overload: the int64 col_indices copy written
    # into out must be identical for every per-dtype value range of the storage.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())
    expected_shape = _expected_col_shape(case)
    out = torch.empty(expected_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(expected_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, expected_shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_empty_bsr(dtype):
    # nnz == 0 for BSR: col and values are empty, but col_indices_copy must
    # still return a (0,) contiguous int64 tensor (not a dense or
    # wrongly-shaped tensor).
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (0,))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_empty_bsr(dtype):
    inp = _make_input("bsr", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(0, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(0, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (0,))


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (cols[0] == cols[1] in row 0), which
    # leaves the CSR tensor with repeated entries; col_indices_copy must still
    # return exactly the stored col array, in storage order, as an independent
    # copy (never a coalesced/sorted array and never an alias). Row 0 holds 3
    # entries for columns [0, 0, 2], so a coalescing implementation would
    # visibly change the stored structure.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_uncoalesced(dtype):
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_col_indices_copy_nan_inf_values(dtype):
    # nan / +-inf stored values must not perturb the returned col_indices
    # copy: col_indices_copy reads only the compressed-index storage, so the
    # copy must still be bit-exact even when the values contain non-finite
    # entries.
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (7,))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_col_indices_copy_out_nan_inf_values(dtype):
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(7, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(7, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (7,))


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_dense():
    # col_indices_copy is a row-compressed-sparse-only metadata accessor: a
    # dense tensor has no compressed column index storage, so both the
    # reference and the candidate must reject it.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_csc():
    # SparseCsc is a distinct compressed layout from SparseCsr: a CSC tensor
    # stores compressed column pointers instead of row pointers and must be
    # rejected.
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    values = torch.randn(4, dtype=torch.float32)
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_coo():
    # Sparse (COO) is a distinct backend key from SparseCsr; col_indices has
    # no Sparse implementation and must reject COO tensors. The reference
    # primary raises NotImplementedError, and the native-body fallback
    # (col_indices) raises RuntimeError, so both are covered here.
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        _reference_col_indices_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_dense():
    # The .out variant is equally row-compressed-sparse-only: a dense tensor
    # has no compressed column index storage and must be rejected.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_csc():
    # The .out variant is equally row-compressed-sparse-only: a CSC tensor is
    # rejected.
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    values = torch.randn(4, dtype=torch.float32)
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_coo():
    # The .out variant is equally row-compressed-sparse-only: a COO tensor is
    # rejected (the reference primary raises NotImplementedError, and the
    # native-body fallback raises RuntimeError).
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_wrong_dtype():
    # The .out contract materializes int64 entries into out; an out tensor of a
    # different dtype must be rejected (ATen raises RuntimeError, and the
    # manual fallback enforces the same contract up front).
    inp = _make_input("csr", (5, 4), 6, None, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(6, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(6, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_col_indices_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
