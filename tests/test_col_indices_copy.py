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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::col_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (nnz,) int64 column index tensor of a sparse row-compressed
# tensor (CSR or BSR) as a fresh, contiguous, independent copy. It is the
# view_copy counterpart of aten::col_indices, whose native body is
# ``col_indices(self).clone(contiguous)``.
#
# The operator is a sparse-metadata accessor, so the regular-operator
# dimensions are adapted as follows:
#   * shapes (tu.selected_shapes): every required rank is mapped onto a CSR
#     layout the operator accepts -- rank 0/1 -> 2-D, rank >= 2 -> batched CSR
#     with the leading dims as batch dims -- plus dedicated CSR/BSR layout
#     cases (single row/column, square, full, empty nnz == 0, ranks 2-5 and
#     block shapes that do not divide the matrix).
#   * value ranges (tu.selected_ranges): the returned col copy never depends on
#     the stored values, so every per-dtype range -- including the extreme
#     [min, 0] / [0, max] magnitudes -- must be accepted and must not perturb
#     the returned array.
#   * dtypes: every storage dtype the sparse CSR/BSR runtime accepts, probed
#     with tu.supported_dtypes (int8 / uint8 / both fp8 formats included when
#     the device supports them), not guessed.
#   * edge cases: empty (nnz == 0, CSR and BSR), uncoalesced duplicate column
#     entries, and nan / +-inf stored values (all ignored by the accessor).
#   * negative: dense, CSC, COO tensors, non-tensor inputs and a wrong-dtype
#     ``out`` tensor are rejected.
#   * broadcast / backward: not applicable -- the operator is unary and returns
#     a fresh int64 metadata tensor, so there is nothing to broadcast against
#     or to differentiate.
#
# Copy semantics are checked on every case: the result must equal the raw col
# array, must be a fresh contiguous int64 tensor that does NOT alias the
# input's internal col storage, and the input must not be mutated.

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
    ("bsr", (4, 6), 0, (2, 2)),
    ("bsr_batch", (2, 4, 6), 6, (2, 2)),
]

# Higher-rank / wider layouts for the "all" level (default, no --quick):
# multi-batch-dim CSR, a BSR whose blocks do not divide the matrix, and a
# batched BSR with a bigger block.
_COLS_ALL = [
    ("csr_batch", (7, 3, 12, 4, 5), 48, None),
    ("bsr", (10, 10), 12, (3, 4)),
    ("bsr_batch", (2, 8, 12), 12, (4, 4)),
]


def _col_cases():
    """(layout, size, nnz, blocks) cases for the quick vs full (default) level."""
    if tu.LEVEL == "quick":
        return [("csr_batch", (2, 19, 7), 20, None)]
    return _COLS_CORE + _COLS_ALL


def _col_value_range_cases():
    """Representative sparse + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [("csr", (5, 4), 6, None)]
    return [
        ("csr", (5, 4), 6, None),
        ("csr_batch", (2, 6, 8), 12, None),
        ("bsr", (4, 6), 4, (2, 2)),
    ]


def _shape_to_csr_case(shape):
    """Map one tu.selected_shapes() rank onto a CSR workload the op accepts.

    Ranks 0/1 become a 2-D CSR matrix (a sparse row-compressed tensor needs at
    least two dims); ranks >= 2 keep the trailing two dims as (rows, cols) and
    use the leading dims as batch dims. The stored nnz is kept tiny so the case
    stays cheap even for logically large shapes.
    """
    if len(shape) == 0:
        size = (1, 1)
    elif len(shape) == 1:
        size = (shape[0], 1)
    else:
        size = tuple(shape)
    rows, cols = size[-2], size[-1]
    nnz = min(rows * cols, 16)
    if len(size) == 2:
        return ("csr", size, nnz, None)
    return ("csr_batch", size, nnz, None)


def _spec_shape_cases():
    """The regular-operator spec's seven shapes, mapped onto CSR workloads."""
    return [_shape_to_csr_case(shape) for shape in tu.selected_shapes()]


def _dedup(dtypes):
    result = []
    for dtype in dtypes:
        if dtype is not None and dtype not in result:
            result.append(dtype)
    return result


def _probe_sparse_storage_dtype(operator, dtype):
    """Probe callable for tu.supported_dtypes.

    A storage dtype is supported when a sparse CSR tensor of that dtype can be
    built on the active device and the ATen metadata accessor materializes its
    row-compressed column index array. Any exception means "unsupported".
    """
    try:
        crow = torch.tensor([0, 2, 3, 5], dtype=torch.long, device=flag_gems.device)
        col = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long, device=flag_gems.device)
        values = torch.ones(5, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_csr_tensor(
            crow, col, values, (3, 4), device=flag_gems.device
        )
        packet = getattr(torch.ops.aten, operator, None)
        if packet is None:
            return False
        out = packet.default(inp)
        return out.dtype == torch.int64 and out.numel() == 5
    except Exception:
        return False


# Candidate storage dtypes: the spec-required nine (int8 / uint8 / fp8 e4m3fn /
# fp8 e5m2 / fp32 / bf16 / fp16 / int32 / int64) plus the remaining families
# the sparse runtime commonly supports (int16 / fp64 / bool). fp8 names are
# looked up defensively for older PyTorch builds.
_CANDIDATE_DTYPES = _dedup(
    tu.REQUIRED_DTYPES
    + [
        torch.int16,
        torch.float64,
        torch.bool,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    ]
)

# Only the dtypes the active device's sparse CSR/BSR runtime + accessor accept;
# probed rather than guessed so unsupported vendor dtypes are skipped. If the
# probe yields nothing, keep the full candidate list rather than a float32-only
# fallback, so a failed/absent probe never silently drops the spec-required
# int8/uint8/fp8 dtypes.
_COLS_DTYPES = tu.supported_dtypes(
    "col_indices_copy",
    candidates=_CANDIDATE_DTYPES,
    probe=_probe_sparse_storage_dtype,
) or list(_CANDIDATE_DTYPES)

# Value-range coverage uses non-bool storage dtypes (bool ignores the range and
# adds nothing beyond the copy-semantics cases above).
_VALUE_RANGE_DTYPES = [dtype for dtype in _COLS_DTYPES if dtype != torch.bool]

# nan / +-inf stored values: only the float dtypes that can represent them.
_NAN_INF_DTYPES = [
    dtype
    for dtype in _COLS_DTYPES
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
]


def _random_crow(n_compressed, nnz, gen):
    # A valid compressed-row array: length n_compressed + 1, non-decreasing,
    # crow[0] == 0 and crow[-1] == nnz. With n_compressed == 1 the array is the
    # degenerate [0, nnz]. Repeated split points leave empty rows, which is
    # valid for the compressed format.
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
    )


def _random_crow_batch(n_batch, n_compressed, nnz, gen):
    # Vectorized batched version of _random_crow: (n_batch, n_compressed + 1),
    # each row non-decreasing with crow[:, 0] == 0 and crow[:, -1] == nnz.
    row0 = torch.zeros(n_batch, 1, dtype=torch.long)
    rown = torch.full((n_batch, 1), nnz, dtype=torch.long)
    if n_compressed == 1:
        return torch.cat([row0, rown], dim=1)
    inner = torch.randint(
        0, nnz + 1, (n_batch, n_compressed - 1), dtype=torch.long, generator=gen
    )
    inner, _ = torch.sort(inner, dim=1)
    return torch.cat([row0, inner, rown], dim=1)


def _make_values(dtype, values_shape, value_range, gen):
    # Stored values come from the shared value-range framework
    # (tu.make_input), which already creates the tensor on flag_gems.device;
    # the returned col copy never depends on them, so every per-dtype range can
    # be exercised through this one constructor. bool ignores the range
    # (deterministic random 0/1 keeps construction reproducible).
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen).to(
            flag_gems.device
        )
    # Values are irrelevant to the returned metadata, so a range that clamps to
    # a single representable value for this dtype (e.g. uint8 with [-1, 0]) is
    # materialized as a constant instead of handing make_tensor a degenerate
    # interval (torch.randint rejects from == to).
    if not (dtype.is_floating_point or dtype.is_complex):
        lo_bound, hi_bound = tu.dtype_bounds(dtype)
        low = int(min(max(tu.resolve_bound(value_range[0], dtype), lo_bound), hi_bound))
        high = int(
            min(max(tu.resolve_bound(value_range[1], dtype), lo_bound), hi_bound)
        )
        if low == high:
            return torch.full(values_shape, low, device=flag_gems.device, dtype=dtype)
    return tu.make_input(dtype, values_shape, list(value_range))


def _make_csr(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    crow = _random_crow(n_rows, nnz, gen)
    col = torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)


def _make_csr_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    n_batch = math.prod(batch_dims)
    crow = _random_crow_batch(n_batch, n_rows, nnz, gen)
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
    # ceil keeps the compressed extents valid for blocks that do not divide the
    # matrix dims; torch.sparse_bsr_tensor infers the block size from the
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
    n_batch = math.prod(batch_dims)
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    crow = _random_crow_batch(n_batch, n_row_blocks, nnz, gen)
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
    if layout in ("csr", "bsr"):
        return (nnz,)
    return size[:-2] + (nnz,)


def _reference_col_indices_copy(inp):
    # The literal ATen operator is the reference (and the expected values all
    # come from it). Probed on CUDA: it is invocable on sparse CSR/BSR tensors
    # for every storage dtype used here, so no composed simulation is needed.
    return torch.ops.aten.col_indices_copy(inp)


def _reference_col_indices_copy_out(inp, out):
    # The .out contract materializes int64 entries into out and returns out
    # itself; ATen enforces the int64 out dtype, so the reference calls the
    # real .out overload directly.
    return torch.ops.aten.col_indices_copy.out(inp, out=out)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins.
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
    # Copy semantics: fresh storage, never a view of the input's col array.
    # For nnz == 0 the result and the input's internal col storage are both
    # empty (data_ptr() == 0), so the pointer check only applies to non-empty
    # results.
    if res.numel() > 0:
        assert res.data_ptr() != inp.col_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot.
    # equal_nan=True keeps the non-mutation check valid for inputs whose stored
    # values contain nan / +-inf.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _col_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


# The .out buffers are garbage-prefilled rather than torch.empty: the .out
# overload must overwrite every element, and torch.empty can hand back a
# recycled allocator block that still holds the expected values, which would
# let a candidate that never writes into out pass. No index array contains
# -1, so it is a safe sentinel.
def _out_buffer(shape, dtype, device):
    # torch.full needs a sequence size, unlike torch.empty which also accepts a
    # bare int.
    if isinstance(shape, int):
        shape = (shape,)
    return torch.full(shape, -1, dtype=dtype, device=device)


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _col_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = _out_buffer(_expected_col_shape(case), torch.long, inp.device)
    ref_out = _out_buffer(_expected_col_shape(case), torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _spec_shape_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_spec_shapes(case, dtype):
    # Shape-level coverage: every tu.selected_shapes() rank (0~5 dims) mapped
    # onto a CSR workload the operator accepts. The logically large shapes keep
    # a tiny nnz, so only the (nnz,) result shape (times the batch size)
    # matters.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _spec_shape_cases())
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = _out_buffer(_expected_col_shape(case), torch.long, inp.device)
    ref_out = _out_buffer(_expected_col_shape(case), torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _col_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_value_ranges(case, value_range, dtype):
    # Value-range coverage: the metadata output (the fresh col int64 copy) is
    # independent of the stored values, so every per-dtype range -- including
    # the extreme [min, 0] and [0, max] magnitudes -- must be accepted and must
    # not perturb the returned column index array.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _col_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_col_indices_copy_out_value_ranges(case, value_range, dtype):
    # Same sweep through the .out overload: the int64 col copy written into out
    # must be identical for every per-dtype value range of the storage.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())
    out = _out_buffer(_expected_col_shape(case), torch.long, inp.device)
    ref_out = _out_buffer(_expected_col_shape(case), torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_col_shape(case))


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
    out = _out_buffer(0, torch.long, inp.device)
    ref_out = _out_buffer(0, torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (0,))


def _uncoalesced_csr(dtype):
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
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_uncoalesced(dtype):
    inp = _uncoalesced_csr(dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out_uncoalesced(dtype):
    inp = _uncoalesced_csr(dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = _out_buffer(5, torch.long, inp.device)
    ref_out = _out_buffer(5, torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


def _nan_inf_csr(dtype):
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
    return torch.sparse_csr_tensor(crow, cols, values, shape)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_col_indices_copy_nan_inf_values(dtype):
    # nan / +-inf stored values must not perturb the returned col copy:
    # col_indices_copy reads only the compressed-index storage, so the copy
    # must still be bit-exact even when the values contain non-finite entries.
    inp = _nan_inf_csr(dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (7,))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_col_indices_copy_out_nan_inf_values(dtype):
    inp = _nan_inf_csr(dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = _out_buffer(7, torch.long, inp.device)
    ref_out = _out_buffer(7, torch.long, ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (7,))


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_dense():
    # col_indices_copy is a row-compressed-sparse-only metadata accessor: a
    # dense tensor has no compressed column index storage, so both the
    # reference and the candidate must reject it.
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_csc():
    # SparseCsc is a distinct compressed layout from SparseCsr: a CSC tensor
    # stores compressed column pointers instead of row pointers and must be
    # rejected.
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_coo():
    # Sparse (COO) is a distinct backend key from SparseCsr; col_indices has no
    # Sparse implementation and must reject COO tensors.
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_col_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.col_indices_copy
def test_col_indices_copy_negative_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        _reference_col_indices_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_dense():
    # The .out variant is equally row-compressed-sparse-only.
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_csc():
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values, (4, 2), device=flag_gems.device
    )
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_col_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.col_indices_copy_out
def test_col_indices_copy_out_negative_wrong_dtype():
    # The .out contract materializes int64 entries into out; an out tensor of a
    # different dtype must be rejected.
    inp = _make_input("csr", (5, 4), 6, None, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(6, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(6, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_col_indices_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
