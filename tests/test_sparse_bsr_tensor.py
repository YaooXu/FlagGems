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

import sys as _sys
from pathlib import Path as _Path

# pytest --import-mode=importlib imports this module as <pkg>.test_sparse_bsr_tensor,
# where <pkg> is the "tests" or "benchmark" package of the checkout that
# actually holds this file (the KernelGen verification harness stages a temp
# copy of the FlagGems tree). When the driving process also has a same-named
# package on sys.path (e.g. the KernelGen repo's own tests/ directory), a bare
# relative import below would bind to that foreign package instead. Put the
# checkout root of *this* file first in sys.path so the relative imports
# resolve to the support files (accuracy_utils/test_utils/base/consts) that
# ship next to it.
_CHECKOUT_ROOT = _Path(__file__).resolve().parent.parent
if str(_CHECKOUT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CHECKOUT_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::sparse_bsr_tensor.crow_col_value_size(Tensor crow_indices,
#     Tensor col_indices, Tensor values, int[] size, *,
#     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=False) -> Tensor constructs a sparse BSR tensor of the
# given ``size`` whose (rows, cols) trailing dims are tiled by ``block``.
#
# aten::sparse_bsr_tensor.crow_col_value(Tensor crow_indices,
#     Tensor col_indices, Tensor values, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False)
#     -> Tensor is the size-inferred variant: rows = (len(crow)-1)*block_rows,
#     cols = (max(col)+1)*block_cols.
#
# The two overloads share one public name, and ``torch.ops.aten.sparse_bsr_tensor``
# dispatches between them by argument count (4 args -> size variant, 3 args ->
# inferred variant); the candidate under test is the same public callable, so
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
#   values verbatim, so the value-range dimension is covered by running the
#   shared tu.selected_ranges() (per-dtype bounds, sign coverage, constants)
#   over the storage values of every float/exact dtype; a dedicated boundary
#   case pins the exact finfo min/max/zero round-trip.
# - Shape levels: tu.selected_shapes() is covered through _shape_level_cases()
#   (which skips the 0-dim scalar, meaningless for a 2-D sparse layout, and
#   turns 1-dim entries into square 2-D tensors), plus dedicated 2-D, batched
#   (3-D/4-D) and empty-grid grids.
# - Broadcast: N/A -- a constructor with three index/value tensors, no
#   broadcasting semantics.
# - Backward: N/A -- the op is a structural constructor with no autograd
#   formula (sparse BSR constructors are non-differentiable).
# - Negative cases: dtype kwarg contradicting the values dtype, missing dtype
#   for non-float32 values, cross-device index/value tensors, and a missing
#   device kwarg on CUDA must raise on the aten reference and the candidate
#   alike.
# - nan/inf: non-finite block values are stored verbatim and compared with
#   equal_nan=True.

# Each 2-D case is (size, block, crow_indices, col_indices): 2x2 square
# blocks, ragged rows with an empty row-block, non-square blocks, and empty
# trailing row-blocks. col entries are always valid (< size[-1] // block[1]).
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
# rows = (len(crow)-1)*br, cols = (max(col)+1)*bc. This overload's inference
# for batched (3-D) values is underspecified, so only 2-D values are covered.
_BSR_2D_INFERRED_CASES = [
    ((2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((2, 3), [0, 1, 2], [0, 1]),
]

# The op copies the given values verbatim into the new tensor (no arithmetic),
# so every float storage dtype the runtime supports is fair game, and exact
# equality holds for integer/bool storages.
_FLOAT_BSR_DTYPES = utils.FLOAT_DTYPES
_EXACT_BSR_DTYPES = utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Block used when deriving grids from the shared shape-level set.
_BLOCK = (2, 2)


def _shape_level_cases():
    # Derive valid (size, block, crow, col) grids from tu.selected_shapes() so
    # the shared shape-level set is covered. The 0-dim scalar entry has no BSR
    # meaning; 1-dim entries become square 2-D tensors. Rows/cols are snapped
    # down to a multiple of the block, and the col grid is a deterministic
    # ragged pattern (every row-block has 1 or 2 blocks, so the grid always
    # has real content).
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


def _make_value_input(shape, dtype, value_range):
    # tu.make_input delegates to torch.testing.make_tensor, which for uint8
    # clamps negative bounds to 0 and then raises on the degenerate randint
    # range; none of the dtypes used here is uint8, so the shared helper is
    # used directly for the spec's per-dtype value ranges.
    return tu.make_input(dtype, shape, value_range)


def _make_bsr_values(nnz, block, dtype, batch=None, value_range=("-1", "1")):
    # Deterministic CPU-side generation of the stored block values so the exact
    # structural copy semantics can be asserted; the tensor is moved to the test
    # device. Shape is (nnz, br, bc), or (batch..., nnz, br, bc) for batched
    # BSR (batch may be an int batch count or a tuple of batch dims).
    # ``value_range`` is a [low, high] symbol pair resolved per-dtype by
    # tu.resolve_bound (the value-range dimension of the regular-operator
    # spec); the data path is a pure copy, so any in-range values round-trip.
    if batch is None:
        shape = (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        shape = batch + (nnz, block[0], block[1])
    else:
        shape = (batch,) + (nnz, block[0], block[1])
    return _make_value_input(shape, dtype, value_range)


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


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_bsr_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_bsr_tensor", getattr(flag_gems, "sparse_bsr_tensor", None)
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
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
    # Block values are stored verbatim, so the float comparison is exact within
    # tolerance; the index arrays must match bit-for-bit.
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())
    # The constructor reads its inputs; it must not mutate them.
    utils.gems_assert_equal(crow_t, ref_crow)
    utils.gems_assert_equal(col_t, ref_col)
    utils.gems_assert_equal(values, ref_values)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_BATCHED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
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
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
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
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_CASES)
@pytest.mark.parametrize("dtype", _EXACT_BSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size_exact_dtypes(case, dtype, value_range):
    # Integer and bool storages: the block values are transferred verbatim, so
    # the comparison is exact (no tolerance).
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
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
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
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _EXACT_BSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_exact_dtypes(case, dtype, value_range):
    # Integer/bool storage through the size-inferred overload.
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
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_shape_levels(case, dtype, value_range):
    # Shape-level dimension: grids derived from tu.selected_shapes() (0-dim
    # scalar excluded; 1-dim entries become square 2-D tensors, all others
    # keep their leading batch dims).
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
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
def test_sparse_bsr_tensor_nan_inf_values(dtype):
    # The nan/inf dimension: non-finite block values are stored verbatim (no
    # arithmetic touches them). Compare with equal_nan=True so nan positions
    # match and the inf signs agree exactly.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    values = _make_bsr_values(nnz, block, dtype)
    values = values.clone()
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
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_equal(res_out.values(), ref_out.values(), equal_nan=True)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
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
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.values(), ref_out.values())


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
    not torch.cuda.is_available(),
    reason="cross-device construction requires a CUDA device",
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
    not torch.cuda.is_available(),
    reason="the missing-device quirk only exists on CUDA",
)
def test_sparse_bsr_tensor_rejects_missing_device():
    # On CUDA this torch build cannot infer the device from the input tensors
    # ("Values and compressed tensor instance need to be on the same device")
    # unless the device kwarg is given; the candidate must reject the same
    # request.
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
