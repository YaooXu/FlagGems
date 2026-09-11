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

# aten::ccol_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (n_cols + 1,) (CSC) / batch_dims + (n_col_blocks + 1,) (BSC)
# int64 compressed-column index tensor of a sparse column-compressed tensor as
# a fresh, contiguous, independent copy. It is the view_copy counterpart of
# aten::ccol_indices, whose native body is
# ``ccol_indices(self).clone(contiguous)``.
#
# The operator is a sparse-metadata accessor, so the regular-operator
# dimensions are adapted as follows:
#   * shapes (tu.selected_shapes): every required rank is mapped onto a CSC
#     layout the operator accepts -- rank 0/1 -> 2-D, rank >= 2 -> batched CSC
#     with the leading dims as batch dims -- plus dedicated CSC/BSC layout
#     cases (single row/column, square, full, empty nnz == 0, ranks 2-5 and
#     block shapes that do not divide the matrix).
#   * value ranges (tu.selected_ranges): the returned ccol copy never depends
#     on the stored values, so every per-dtype range -- including the extreme
#     [min, 0] / [0, max] magnitudes -- must be accepted and must not perturb
#     the returned array.
#   * dtypes: every storage dtype the sparse CSC/BSC runtime accepts, probed
#     with tu.supported_dtypes (int8 / uint8 / both fp8 formats included when
#     the device supports them), not guessed.
#   * edge cases: empty (nnz == 0), uncoalesced duplicate row entries, and
#     nan / +-inf stored values (all ignored by the accessor).
#   * negative: dense, CSR, COO tensors and a wrong-dtype ``out`` are rejected.
#   * broadcast / backward: not applicable -- the operator is unary and returns
#     a fresh int64 metadata tensor, so there is nothing to broadcast against
#     or to differentiate.
#
# Copy semantics are checked on every case: the result must equal the raw ccol
# array, must be a fresh contiguous int64 tensor that does NOT alias the
# input's internal ccol storage, and the input must not be mutated.

# (layout, size, nnz, blocks) core cases: 2-D CSC (incl. single-row/single-
# column/square/nnz==0), batched CSC, 2-D BSC with varied block shapes, batched
# BSC, and an empty BSC.
_CCOLS_CORE = [
    ("csc", (5, 4), 6, None),
    ("csc", (4, 1), 3, None),
    ("csc", (1, 5), 2, None),
    ("csc", (8, 8), 16, None),
    ("csc", (16, 32), 40, None),
    ("csc", (32, 16), 80, None),
    ("csc", (3, 3), 9, None),
    ("csc", (3, 4), 0, None),
    ("csc_batch", (2, 6, 8), 12, None),
    ("bsc", (4, 6), 4, (2, 2)),
    ("bsc", (8, 8), 8, (2, 2)),
    ("bsc", (6, 6), 6, (3, 2)),
    ("bsc", (4, 6), 0, (2, 2)),
    ("bsc_batch", (2, 4, 6), 6, (2, 2)),
]

# Higher-rank / wider layouts for the "all" level (default, no --quick):
# multi-batch-dim CSC, a BSC whose column blocks do not divide the column
# count, and a batched BSC with a bigger block.
_CCOLS_ALL = [
    ("csc_batch", (7, 3, 12, 4, 5), 48, None),
    ("bsc", (10, 10), 12, (3, 4)),
    ("bsc_batch", (2, 8, 12), 12, (4, 4)),
]


def _ccol_cases():
    """(layout, size, nnz, blocks) cases for the quick vs full (default) level."""
    if tu.LEVEL == "quick":
        return [("csc_batch", (2, 19, 7), 20, None)]
    return _CCOLS_CORE + _CCOLS_ALL


def _ccol_value_range_cases():
    """Representative sparse + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [("csc", (5, 4), 6, None)]
    return [
        ("csc", (5, 4), 6, None),
        ("csc_batch", (2, 6, 8), 12, None),
        ("bsc", (4, 6), 4, (2, 2)),
    ]


def _shape_to_csc_case(shape):
    """Map one tu.selected_shapes() rank onto a CSC/BSC layout the op accepts.

    Ranks 0/1 become a 2-D CSC matrix (a sparse column-compressed tensor needs
    at least two dims); ranks >= 2 keep the trailing two dims as (rows, cols)
    and use the leading dims as batch dims. The stored nnz is kept tiny so the
    case stays cheap even for logically large shapes.
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
        return ("csc", size, nnz, None)
    return ("csc_batch", size, nnz, None)


def _spec_shape_cases():
    """The regular-operator spec's seven shapes, mapped onto CSC workloads."""
    return [_shape_to_csc_case(shape) for shape in tu.selected_shapes()]


def _dedup(dtypes):
    result = []
    for dtype in dtypes:
        if dtype is not None and dtype not in result:
            result.append(dtype)
    return result


def _probe_sparse_storage_dtype(operator, dtype):
    """Probe callable for tu.supported_dtypes.

    A storage dtype is supported when a sparse CSC tensor of that dtype can be
    built on the active device and the ATen metadata accessor materializes its
    compressed-column array. Any exception means "unsupported".
    """
    try:
        ccol = torch.tensor([0, 2, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
        row = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long, device=flag_gems.device)
        values = torch.ones(5, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_csc_tensor(
            ccol, row, values, (3, 4), device=flag_gems.device
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

# Only the dtypes the active device's sparse CSC/BSC runtime + accessor accept;
# probed rather than guessed so unsupported vendor dtypes are skipped. If the
# probe yields nothing, keep the full candidate list rather than a float32-only
# fallback, so a failed/absent probe never silently drops the spec-required
# int8/uint8/fp8 dtypes.
_CCOLS_DTYPES = tu.supported_dtypes(
    "ccol_indices_copy",
    candidates=_CANDIDATE_DTYPES,
    probe=_probe_sparse_storage_dtype,
) or list(_CANDIDATE_DTYPES)

# Value-range coverage uses non-bool storage dtypes (bool ignores the range and
# adds nothing beyond the copy-semantics cases above).
_VALUE_RANGE_DTYPES = [dtype for dtype in _CCOLS_DTYPES if dtype != torch.bool]

# nan / +-inf stored values: only the float dtypes that can represent them.
_NAN_INF_DTYPES = [
    dtype
    for dtype in _CCOLS_DTYPES
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
]


def _random_ccol(n_compressed, nnz, gen):
    # A valid compressed-column array: length n_compressed + 1, non-decreasing,
    # ccol[0] == 0 and ccol[-1] == nnz. With n_compressed == 1 the array is the
    # degenerate [0, nnz].
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
    )


def _random_ccol_batch(n_batch, n_compressed, nnz, gen):
    # Vectorized batched version of _random_ccol: (n_batch, n_compressed + 1),
    # each row non-decreasing with ccol[:, 0] == 0 and ccol[:, -1] == nnz.
    col0 = torch.zeros(n_batch, 1, dtype=torch.long)
    coln = torch.full((n_batch, 1), nnz, dtype=torch.long)
    if n_compressed == 1:
        return torch.cat([col0, coln], dim=1)
    inner = torch.randint(
        0, nnz + 1, (n_batch, n_compressed - 1), dtype=torch.long, generator=gen
    )
    inner, _ = torch.sort(inner, dim=1)
    return torch.cat([col0, inner, coln], dim=1)


def _make_values(dtype, values_shape, value_range, gen):
    # Stored values come from the shared value-range framework
    # (tu.make_input), which already creates the tensor on flag_gems.device;
    # the returned ccol copy never depends on them, so every per-dtype range
    # can be exercised through this one constructor. bool ignores the range
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


def _make_csc(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    ccol = _random_ccol(n_cols, nnz, gen)
    rows = torch.randint(0, n_rows, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csc_tensor(ccol, rows, values, size=size, device=device)


def _make_csc_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    n_batch = math.prod(batch_dims)
    ccol = _random_ccol_batch(n_batch, n_cols, nnz, gen)
    rows = torch.randint(0, n_rows, (n_batch, nnz), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (n_batch, nnz), value_range, gen)
    return torch.sparse_csc_tensor(
        ccol.view(batch_dims + (n_cols + 1,)),
        rows.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz,)),
        size=size,
        device=device,
    )


def _make_bsc(size, nnz, blocks, dtype, gen, device, value_range):
    n_rows, n_cols = size
    block_rows, block_cols = blocks
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    ccol = _random_ccol(n_col_blocks, nnz, gen)
    row = torch.randint(0, n_row_blocks, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range, gen)
    # torch.sparse_bsc_tensor infers the block size from the trailing dims of
    # the values tensor (values_shape == (nnz, block_rows, block_cols)).
    return torch.sparse_bsc_tensor(ccol, row, values, size=size, device=device)


def _make_bsc_batch(size, nnz, blocks, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    block_rows, block_cols = blocks
    n_batch = math.prod(batch_dims)
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    ccol = _random_ccol_batch(n_batch, n_col_blocks, nnz, gen)
    row = torch.randint(
        0, n_row_blocks, (n_batch, nnz), dtype=torch.long, generator=gen
    )
    values = _make_values(
        dtype, (n_batch, nnz, block_rows, block_cols), value_range, gen
    )
    return torch.sparse_bsc_tensor(
        ccol.view(batch_dims + (n_col_blocks + 1,)),
        row.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz, block_rows, block_cols)),
        size=size,
        device=device,
    )


def _make_input(layout, size, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    # Deterministic CPU-side compressed/row index generation; the values tensor
    # comes from the shared value-range helper (tu.make_input) and the sparse
    # tensor is created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csc":
        return _make_csc(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "csc_batch":
        return _make_csc_batch(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "bsc":
        return _make_bsc(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)
    return _make_bsc_batch(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)


def _expected_ccol_shape(case):
    layout, size, nnz, blocks = case
    del nnz
    if layout == "csc":
        return (size[-1] + 1,)
    if layout == "csc_batch":
        return size[:-2] + (size[-1] + 1,)
    n_col_blocks = int(math.ceil(size[-1] / blocks[1]))
    if layout == "bsc":
        return (n_col_blocks + 1,)
    return size[:-2] + (n_col_blocks + 1,)


def _reference_ccol_indices_copy(inp):
    # The literal ATen operator is the reference (and the expected values all
    # come from it). Probed on CUDA and CPU: it is invocable on sparse
    # CSC/BSC tensors for every storage dtype used here, so no composed
    # simulation is needed.
    return torch.ops.aten.ccol_indices_copy(inp)


def _reference_ccol_indices_copy_out(inp, out):
    # The .out contract materializes int64 entries into out and returns out
    # itself; ATen enforces the int64 out dtype, so the reference calls the
    # real .out overload directly.
    return torch.ops.aten.ccol_indices_copy.out(inp, out=out)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins.
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices_copy", getattr(flag_gems, "ccol_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices_copy.out", getattr(flag_gems, "ccol_indices_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_shape):
    # ccol_indices_copy returns a fresh contiguous int64 tensor holding the
    # input's raw compressed-column array. The result must not alias the
    # input's internal ccol storage and the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == expected_shape
    assert ref.shape == expected_shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's ccol array.
    assert res.data_ptr() != inp.ccol_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot.
    # equal_nan=True keeps the non-mutation check valid for inputs whose stored
    # values contain nan / +-inf.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _ccol_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _ccol_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(_expected_ccol_shape(case), dtype=torch.long, device=inp.device)
    ref_out = torch.empty(
        _expected_ccol_shape(case), dtype=torch.long, device=ref_inp.device
    )

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _spec_shape_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_spec_shapes(case, dtype):
    # Shape-level coverage: every tu.selected_shapes() rank (0~5 dims) mapped
    # onto a CSC workload the operator accepts. The logically large shapes keep
    # a tiny nnz, so only the compressed extent (n_cols + 1, times the batch
    # size) matters.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _spec_shape_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out_spec_shapes(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(_expected_ccol_shape(case), dtype=torch.long, device=inp.device)
    ref_out = torch.empty(
        _expected_ccol_shape(case), dtype=torch.long, device=ref_inp.device
    )

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _ccol_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_ccol_indices_copy_value_ranges(case, value_range, dtype):
    # Value-range coverage: the metadata output (the fresh ccol int64 copy) is
    # independent of the stored values, so every per-dtype range -- including
    # the extreme [min, 0] and [0, max] magnitudes -- must be accepted and must
    # not perturb the returned ccol array.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _ccol_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_ccol_indices_copy_out_value_ranges(case, value_range, dtype):
    # Same sweep through the .out overload: the int64 ccol copy written into
    # out must be identical for every per-dtype value range of the storage.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(_expected_ccol_shape(case), dtype=torch.long, device=inp.device)
    ref_out = torch.empty(
        _expected_ccol_shape(case), dtype=torch.long, device=ref_inp.device
    )

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_empty_bsc(dtype):
    # nnz == 0 for BSC: rows and values are empty, but ccol_indices_copy must
    # still return a (n_col_blocks + 1,) contiguous int64 tensor.
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (4,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out_empty_bsc(dtype):
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(4, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(4, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (4,))


def _uncoalesced_csc(dtype):
    # The (0, 0) entry is duplicated (rows[0] == rows[1] in column 0), which
    # leaves the CSC tensor with repeated entries; column 0 holds 3 entries for
    # rows [0, 0, 2]. ccol_indices_copy must return exactly the stored ccol
    # array, in storage order, as an independent copy (never coalesced/sorted
    # and never an alias).
    shape = (3, 4)
    ccol = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    return torch.sparse_csc_tensor(ccol, rows, values, shape)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_uncoalesced(dtype):
    inp = _uncoalesced_csc(dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out_uncoalesced(dtype):
    inp = _uncoalesced_csc(dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


def _nan_inf_csc(dtype):
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
    return torch.sparse_csc_tensor(ccol, rows, values, shape)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_ccol_indices_copy_nan_inf_values(dtype):
    # nan / +-inf stored values must not perturb the returned ccol copy:
    # ccol_indices_copy reads only the compressed-index storage, so the copy
    # must still be bit-exact even when the values contain non-finite entries.
    inp = _nan_inf_csc(dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_ccol_indices_copy_out_nan_inf_values(dtype):
    inp = _nan_inf_csc(dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_dense():
    # ccol_indices_copy is a column-compressed-sparse-only metadata accessor: a
    # dense tensor has no compressed column index storage, so both the
    # reference and the candidate must reject it.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_csr():
    # SparseCsr is a distinct compressed layout from SparseCsc: a CSR tensor
    # stores crow/col pointers instead of a compressed column array and must be
    # rejected.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_coo():
    # Sparse (COO) is a distinct backend key from SparseCsr (CSC); ccol_indices
    # has no Sparse implementation and must reject COO tensors.
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_dense():
    # The .out variant is equally column-compressed-sparse-only.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_csr():
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_coo():
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_wrong_dtype():
    # The .out contract materializes int64 entries into out; an out tensor of a
    # different dtype must be rejected.
    inp = _make_input("csc", (5, 4), 6, None, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_ccol_indices_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
