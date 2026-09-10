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

# aten::sparse_bsr_tensor.crow_col_value_size(Tensor crow_indices,
#     Tensor col_indices, Tensor values, int[] size, *,
#     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=False) -> Tensor constructs a sparse BSR tensor of the
# given ``size`` whose trailing (rows, cols) dims are tiled by the block shape
# inferred from ``values``.
#
# aten::sparse_bsr_tensor.crow_col_value(Tensor crow_indices,
#     Tensor col_indices, Tensor values, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False)
#     -> Tensor is the size-inferred variant: rows = (len(crow)-1)*block_rows,
#     cols = (max(col)+1)*block_cols.
#
# The two overloads share one public name and ``torch.ops.aten.sparse_bsr_tensor``
# dispatches between them by argument count (4 args -> size variant, 3 args ->
# inferred variant). The candidate under test is the same public callable, so
# every reference call below mirrors the candidate call exactly. The ``dtype``
# keyword is always passed explicitly: without it the aten op forces the values
# to float32 and raises RuntimeError for any other storage dtype. The ``device``
# keyword is passed explicitly too: on CUDA this torch build fails to infer the
# device from the input tensors ("Values and compressed tensor instance need to
# be on the same device") unless the target device is given.
#
# BSR layout facts exercised below: layout == torch.sparse_bsr, sparse_dim == 2,
# dense_dim == ndim - 2, values shape is (nnz, br, bc) for 2-D tensors and
# (batch, nnz, br, bc) for batched (3-D and higher) tensors, and crow/col hold
# the block grid with col entries in [0, n_col_blocks).
#
# Regular-operator spec dimensions:
# - Value ranges: the data path is a pure copy -- the op stores the given block
#   values verbatim -- so every value range in tu.selected_ranges() is applied
#   to the storage values of every supported dtype; a dedicated boundary case
#   pins the exact finfo min/max/zero round-trip.
# - Shape levels: tu.selected_shapes() is covered through _shape_level_cases()
#   (the 0-dim scalar is skipped as meaningless for a 2-D sparse layout, and
#   1-dim entries become square 2-D tensors), plus dedicated 2-D, batched
#   (3-D/4-D) and empty-grid grids.
# - Broadcast: N/A -- a constructor with three index/value tensors has no
#   broadcasting semantics.
# - Backward: N/A -- the op is a structural constructor with no autograd
#   formula (sparse BSR constructors are non-differentiable).
# - Negative cases: dtype kwarg contradicting the values dtype, missing dtype
#   for non-float32 values, cross-device index/value tensors, and a missing
#   device kwarg on this torch build must raise on the aten reference and the
#   candidate alike.
# - nan/inf: non-finite block values are stored verbatim and compared with
#   equal_nan=True.

# Each 2-D case is (size, block, crow_indices, col_indices): 2x2 square blocks,
# ragged rows with an empty row-block, non-square blocks, and empty trailing
# row-blocks. col entries are always valid (< size[-1] // block[1]).
_BSR_2D_CASES = [
    ((4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((6, 6), (2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((4, 6), (2, 3), [0, 1, 2], [0, 1]),
    ((8, 8), (4, 2), [0, 2, 3], [1, 3, 0]),
    ((6, 4), (3, 2), [0, 1, 2], [0, 1]),
    ((10, 12), (2, 4), [0, 3, 5, 6, 6, 6], [0, 1, 2, 0, 2, 1]),
]

# Each batched case is (size, block, crow_indices, col_indices) with the batch
# dimensions first: (batch..., rows, cols). All batches share one crow/col
# block grid; the per-batch block values live in values[batch..., :, :, :].
# The last entry exercises a 4-D tensor (two batch dims).
_BSR_BATCHED_CASES = [
    ((2, 6, 6), (2, 3), [0, 2, 4, 4], [0, 1, 0, 1]),
    ((3, 4, 8), (2, 4), [0, 2, 3], [0, 1, 0]),
    ((2, 4, 4), (2, 2), [0, 2, 2], [0, 1]),
    ((2, 8, 12), (4, 3), [0, 1, 3], [0, 3, 1]),
    ((2, 3, 4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
]

# Empty storage (nnz == 0): the block grid still exists but stores no blocks.
# Each case is (size, block, batch); batch is None for the 2-D tensor.
_BSR_EMPTY_CASES = [
    ((4, 4), (2, 2), None),
    ((2, 4, 4), (2, 2), 2),
]

# Size-inferred ``crow_col_value`` cases as (block, crow_indices, col_indices)
# with values of shape (nnz, br, bc). Expected size is derived from crow/col:
# rows = (len(crow)-1)*br, cols = (max(col)+1)*bc. This overload's inference for
# batched (3-D) values is underspecified, so only 2-D values are covered.
_BSR_2D_INFERRED_CASES = [
    ((2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((2, 3), [0, 1, 2], [0, 1]),
]

# Block used when deriving grids from the shared shape-level set.
_BLOCK = (2, 2)

# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------
#
# The regular-operator spec requires int8, uint8, the two fp8 formats, fp32,
# bf16, fp16, int32 and int64 where the operator supports them. The BSR factory
# accepts every storage dtype the underlying tensor supports, so the list is
# probed against the real ATen call (tu.supported_dtypes cannot be used here: it
# probes ``packet.default(x)`` with a single tensor, while this op needs three
# component tensors).
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_REQUIRED_VALUE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]
# bool and int16 were part of the pre-existing coverage; float64 is added when
# the backend supports it (part of the shared float set).
_EXTRA_VALUE_DTYPES = [torch.bool, torch.int16, torch.float64]


def _probe_value_dtypes(candidates):
    crow = torch.tensor([0, 1, 1], dtype=torch.long, device=flag_gems.device)
    col = torch.tensor([0], dtype=torch.long, device=flag_gems.device)
    supported = []
    for dtype in candidates:
        try:
            values = torch.zeros((1, 2, 2), dtype=dtype, device=flag_gems.device)
            torch.ops.aten.sparse_bsr_tensor(
                crow, col, values, [2, 2], dtype=dtype, device=flag_gems.device
            )
        except Exception:
            continue
        supported.append(dtype)
    return supported


_VALUE_DTYPES = _probe_value_dtypes(_REQUIRED_VALUE_DTYPES + _EXTRA_VALUE_DTYPES)
if not _VALUE_DTYPES:
    _VALUE_DTYPES = [torch.float32]

# Exact-copy dtypes (integer/bool and the fp8 formats) are asserted bit-exactly
# with gems_assert_equal; true float dtypes use the tolerance-based helper. The
# fp8 formats live here because torch.testing.assert_close has no fp8
# comparison kernel, while the values are copied verbatim anyway.
_EXACT_VALUE_DTYPES = frozenset(
    dtype
    for dtype in _VALUE_DTYPES
    if (not dtype.is_floating_point) or dtype in _FP8_DTYPES
)
_FLOAT_VALUE_DTYPES = [
    dtype
    for dtype in _VALUE_DTYPES
    if dtype.is_floating_point and dtype not in _FP8_DTYPES
] or [torch.float32]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_level_cases():
    # Derive valid (size, block, crow, col) grids from tu.selected_shapes() so
    # the shared shape-level set is covered. The 0-dim scalar entry has no BSR
    # meaning; 1-dim entries become square 2-D tensors. Rows/cols are snapped
    # down to a multiple of the block, and the col grid is a deterministic
    # ragged pattern (every row-block has 1 or 2 blocks, so the grid always has
    # real content).
    cases = []
    for shape in tu.selected_shapes():
        if len(shape) == 0:
            continue
        if len(shape) == 1:
            batch, rows, cols = (), shape[0], shape[0]
        else:
            batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        br, bc = _BLOCK
        rows = max(br, rows // br * br)
        cols = max(bc, cols // bc * bc)
        n_row_blocks = rows // br
        n_col_blocks = cols // bc
        gen = torch.Generator("cpu").manual_seed(len(shape))
        crow = [0]
        col = []
        for r in range(n_row_blocks):
            k = min(1 + (r % 2), n_col_blocks)
            chosen = torch.randperm(n_col_blocks, generator=gen)[:k].sort().values
            col.extend(chosen.tolist())
            crow.append(crow[-1] + k)
        cases.append((batch + (rows, cols), _BLOCK, crow, col))
    return cases


def _make_index_tensor(indices):
    return torch.tensor(indices, dtype=torch.long, device=flag_gems.device)


def _make_values(shape, dtype, value_range):
    # Value-range dimension of the regular-operator spec. Bounds resolve
    # per-dtype (max/min are the dtype limits); unsigned dtypes clip the
    # negative lower bound, and a degenerate (low >= high) range fills the
    # constant instead of calling make_tensor, which rejects an empty interval.
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        low = max(low, finfo.min)
        high = min(high, finfo.max)
        if not low < high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
    else:
        low, high = int(low), int(high)
        dmin, dmax = tu.dtype_bounds(dtype)
        low = max(low, int(dmin))
        high = min(max(high, low), int(dmax))
        if low >= high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _make_bsr_values(nnz, block, dtype, batch=None, value_range=("-1", "1")):
    # Deterministic, contiguous block values of shape (nnz, br, bc), or
    # (batch..., nnz, br, bc) for batched BSR (batch may be an int batch count
    # or a tuple of batch dims).
    if batch is None:
        shape = (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        shape = batch + (nnz, block[0], block[1])
    else:
        shape = (batch,) + (nnz, block[0], block[1])
    return _make_values(shape, dtype, value_range)


def _assert_bsr_structure(out, size, block, nnz, dtype, batch=None):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # sparse/dense split, block grid capacity, and the nnz count.
    assert out.layout == torch.sparse_bsr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == len(size) - 2
    assert out._nnz() == nnz
    if batch is None:
        assert tuple(out.values().shape) == (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        assert tuple(out.values().shape) == batch + (nnz, block[0], block[1])
    else:
        assert tuple(out.values().shape) == (batch,) + (nnz, block[0], block[1])
    n_row_blocks = size[-2] // block[0]
    n_col_blocks = size[-1] // block[1]
    assert len(out.crow_indices()) == n_row_blocks + 1
    assert (out.col_indices() < n_col_blocks).all()
    assert (out.col_indices() >= 0).all()


def _assert_sparse(res, ref, dtype, equal_nan=False):
    if dtype in _EXACT_VALUE_DTYPES:
        utils.gems_assert_equal(res, ref, equal_nan=equal_nan)
    else:
        utils.gems_assert_close(res, ref, dtype, equal_nan=equal_nan)
    utils.gems_assert_equal(res.values(), ref.values(), equal_nan=equal_nan)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_bsr_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_bsr_tensor", getattr(flag_gems, "sparse_bsr_tensor", None)
    )


# ---------------------------------------------------------------------------
# Forward coverage
# ---------------------------------------------------------------------------


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size(case, dtype, value_range):
    size, block, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_sparse(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    # The constructor reads its inputs; it must not mutate them.
    utils.gems_assert_equal(crow_t, ref_crow)
    utils.gems_assert_equal(col_t, ref_col)
    utils.gems_assert_equal(values, ref_values)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_BATCHED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size_batched(case, dtype, value_range):
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    _assert_sparse(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_crow_col_value_size_empty(case, dtype):
    # Empty storage (nnz == 0): the grid still exists but stores no blocks.
    size, block, batch = case
    n_row_blocks = size[-2] // block[0]
    crow_t = torch.zeros(n_row_blocks + 1, dtype=torch.long, device=flag_gems.device)
    col_t = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(0, block, dtype, batch=batch)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, 0, dtype, batch=batch)
    _assert_sparse(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value(case, dtype, value_range):
    # Size-inferred overload: the 3-argument call (no size) derives the tensor
    # size from crow/col and the block shape of the values.
    block, crow, col = case
    nnz = len(col)
    size = ((len(crow) - 1) * block[0], (max(col) + 1) * block[1])
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_sparse(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_shape_levels(case, dtype, value_range):
    # Shape-level dimension: grids derived from tu.selected_shapes() (0-dim
    # scalar excluded; 1-dim entries become square 2-D tensors, all others keep
    # their leading batch dims).
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    _assert_sparse(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_nan_inf_values(dtype):
    # The nan/inf dimension: non-finite block values are stored verbatim (no
    # arithmetic touches them). Compare with equal_nan=True so nan positions
    # match and the inf signs agree exactly.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    values = _make_bsr_values(nnz, block, dtype)
    flat = values.reshape(-1)
    flat[0] = float("nan")
    flat[1] = float("inf")
    flat[2] = float("-inf")
    flat[-1] = float("nan")
    values = flat.reshape(values.shape)
    ref_values = utils.to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = _resolve_gems_op()(
        _make_index_tensor([0, 2, 4]),
        _make_index_tensor([0, 1, 0, 1]),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_sparse(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_boundary_values(dtype):
    # torch.testing.make_tensor draws values strictly inside the dtype bounds,
    # so pin the exact finfo min/max (and a few exact constants) explicitly:
    # the op stores values verbatim, so the boundary values must round-trip
    # bit-exactly.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    finfo = torch.finfo(dtype)
    specials = torch.tensor(
        [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    n_elems = nnz * block[0] * block[1]
    values = specials.repeat((n_elems + specials.numel() - 1) // specials.numel())[
        :n_elems
    ]
    values = values.reshape(nnz, block[0], block[1])
    ref_values = utils.to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = _resolve_gems_op()(
        _make_index_tensor([0, 2, 4]),
        _make_index_tensor([0, 1, 0, 1]),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_sparse(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Negative cases: each invalid request must raise on the aten reference and the
# candidate must reject it too rather than silently succeeding.
# ---------------------------------------------------------------------------


@pytest.mark.sparse_bsr_tensor
def test_sparse_bsr_tensor_rejects_dtype_mismatch():
    # The dtype kwarg must match the values dtype; the aten reference raises
    # RuntimeError ("dtype of values (Half) must match dtype of sparse tensor
    # (Float)") and the candidate must reject the call too.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, torch.float16)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op()(
            crow_t, col_t, values, list(size), dtype=torch.float32, device=crow_t.device
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_sparse_bsr_tensor_rejects_missing_dtype(dtype):
    # Without an explicit dtype the aten op forces float32 and raises for any
    # other storage dtype; the candidate must reject the same request.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow, ref_col, ref_values, list(size), device=ref_crow.device
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op()(crow_t, col_t, values, list(size), device=crow_t.device)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_bsr_tensor_rejects_device_mismatch():
    # All three storage tensors must share one device; the aten reference
    # rejects a values tensor on a different device than crow/col ("Values and
    # crow_indices need to be on the same device").
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, torch.float32)
    cpu_values = values.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op()(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="device inference from component tensors only applies to accelerators",
)
def test_sparse_bsr_tensor_rejects_missing_device():
    # On this torch build the factory cannot infer the device from the input
    # tensors ("Values and compressed tensor instance need to be on the same
    # device") unless the device kwarg is given; the candidate must reject the
    # same request.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op()(crow_t, col_t, values, list(size), dtype=torch.float32)
