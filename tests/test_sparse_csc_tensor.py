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
# earlier on sys.path than this checkout's ``tests`` package. With
# ``--import-mode=importlib`` pytest does not prepend the checkout root, so
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

# aten::sparse_csc_tensor.ccol_row_value_size(Tensor ccol_indices,
#     Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# constructs a sparse CSC tensor from compressed column pointers (length N + 1
# for an N-column matrix), stored row indices (length nnz) and values (shape
# (nnz,) for 2-D or (batch..., nnz) for batched), laid out over a logical
# matrix of size ``size``. The no-size sibling
# (``aten::sparse_csc_tensor.ccol_row_value``) infers the shape from the index
# tensors: rows = max(row) + 1, cols = len(ccol) - 1.
#
# Both overloads share the single public name ``sparse_csc_tensor``; the aten
# dispatcher picks the overload by argument count (4 positional args -> the
# explicit-size variant, 3 -> the shape-inferred variant). The candidate under
# test is the same public callable, so every reference call below mirrors the
# candidate call exactly (same argument order, same keyword set).
#
# The ``dtype`` keyword is always passed explicitly: without it the aten op
# forces float32 storage and raises for every other values dtype ("dtype of
# values (...) must match dtype of sparse tensor"). The ``device`` keyword is
# passed explicitly too: on CUDA this torch build fails to infer the target
# device from the input tensors ("Values and compressed tensor instance need
# to be on the same device") unless it is given. ``layout=torch.sparse_csc``
# pins the compressed-column layout.
#
# CSC layout facts exercised below: layout == torch.sparse_csc, sparse_dim == 2,
# dense_dim == 0 (the batch dims of batched tensors are sparse batch dims),
# values stored at values[batch..., nnz], and ccol/row carry the compressed
# column structure verbatim (the constructor never re-sorts or coalesces).
#
# Regular-operator spec dimensions:
# - Value ranges: the data path of this constructor is a pure copy -- the op
#   stores the given values verbatim -- so the value-range dimension is covered
#   by running the shared tu.selected_ranges() (sign coverage, per-dtype bounds
#   and constants) over the storage values of every supported float/exact
#   dtype, plus a dedicated boundary case pinning the finfo min/max/zero
#   round-trip.
# - Shape levels: tu.selected_shapes() is covered through _shape_level_cases()
#   (which skips the 0-dim scalar, meaningless for a 2-D sparse layout, and
#   turns 1-dim entries into square 2-D tensors), plus dedicated 2-D, batched
#   and empty (nnz == 0) cases.
# - Broadcast: N/A -- a constructor taking three index/value tensors, with no
#   broadcasting semantics.
# - Backward: N/A -- the op is a structural constructor with no autograd
#   formula (sparse CSC constructors are non-differentiable).
# - Negative cases: a dtype kwarg contradicting the values dtype, a missing
#   dtype for non-float32 values, a non-CSC layout kwarg, cross-device
#   index/value tensors, and a missing device kwarg on CUDA must raise on the
#   aten reference and the candidate alike.
# - nan/inf: non-finite values are stored verbatim and compared with
#   equal_nan=True.

# ---------------------------------------------------------------------------
# Shared cases and dtype sets
# ---------------------------------------------------------------------------

# (matrix shape, nnz) pairs: 1-D row, tall, wide, square and empty (nnz == 0).
_CSC_CASES = [
    ((1, 8), 4),
    ((8, 1), 4),
    ((4, 4), 16),
    ((5, 7), 13),
    ((7, 5), 13),
    ((4, 4), 0),
]

# Batched CSC: 3-D and 4-D matrices; batch dims become sparse batch dims of the
# resulting tensor (sparse_dim stays 2, dense_dim stays 0).
_CSC_BATCHED_CASES = [
    ((2, 4, 4), 4),
    ((3, 6, 5), 6),
]

# Shape-inferred (no-size) overload: ccol is given as a Python list (batch
# shape (N + 1,)), row as a Python list of length nnz; expected rows are
# max(row) + 1.
_CSC_NO_SIZE_CASES = [
    ([0, 2, 3, 5], [0, 2, 1, 3, 2], (4, 3)),
    ([0, 3, 3, 4], [0, 1, 2, 3], (4, 3)),
    ([0, 0, 0], [], (0, 2)),
    ([0, 2, 4], [0, 1, 1, 0], (2, 2)),
]

# The op stores every dtype the CUDA storage supports: fp16/fp32/bf16 (plus
# fp64 when the device supports it), int16/int32/int64 and bool. Index tensors
# are int32 or int64.
_FLOAT_CSC_DTYPES = utils.ALL_FLOAT_DTYPES
_EXACT_CSC_DTYPES = utils.ALL_INT_DTYPES + utils.BOOL_TYPES
_CSC_DTYPES = _FLOAT_CSC_DTYPES + _EXACT_CSC_DTYPES
_INDEX_DTYPES = [torch.int32, torch.int64]


def _make_values(nnz, dtype, shape=None, value_range=("-1", "1")):
    # The stored values come from the shared value-range framework: the
    # [low, high] range symbols resolve per-dtype via tu.make_input and the
    # tensor is generated on the test device. The construction is a pure copy,
    # so every in-range value round-trips verbatim.
    shape = (nnz,) if shape is None else shape
    return tu.make_input(dtype, shape, value_range).to(flag_gems.device)


def _make_csc_inputs(
    shape, nnz, dtype, seed=0, index_dtype=torch.int64, value_range=("-1", "1")
):
    """Build valid (ccol_indices, row_indices, values) for a (batch..., M, N)
    logical matrix with ``nnz`` stored entries.

    Indices are generated deterministically on the CPU and moved to the test
    device; values come from the value-range framework. The nnz entries are
    spread across the N column blocks by random cut points, and the row
    indices are drawn with replacement (duplicate entries and unsorted rows
    are legal CSC structure that the constructor must keep verbatim)."""
    device = flag_gems.device
    gen = torch.Generator("cpu").manual_seed(seed)
    batch, (M, N) = shape[:-2], shape[-2:]
    total_batch = 1
    for d in batch:
        total_batch *= d
    if N <= 1:
        counts = torch.full((total_batch, 1), nnz, dtype=torch.long)
    else:
        cuts = torch.sort(
            torch.randint(0, nnz + 1, (total_batch, N - 1), generator=gen)
        ).values
        bounds = torch.cat(
            [
                torch.zeros(total_batch, 1, dtype=torch.long),
                cuts[:, : N - 1],
                torch.full((total_batch, 1), nnz, dtype=torch.long),
            ],
            dim=-1,
        )
        counts = bounds[..., 1:] - bounds[..., :-1]
    ccol = torch.cat(
        [torch.zeros(total_batch, 1, dtype=torch.long), torch.cumsum(counts, -1)],
        dim=-1,
    ).view(batch + (N + 1,))
    row = torch.randint(0, M, (total_batch, nnz), generator=gen).view(batch + (nnz,))
    values = _make_values(nnz, dtype, shape=batch + (nnz,), value_range=value_range)
    return (
        ccol.to(device=device, dtype=index_dtype),
        row.to(device=device, dtype=index_dtype),
        values,
    )


def _resolve_gems_op():
    # resolve_gems_op() picks up a KernelGen-injected override first and only
    # then falls back to flag_gems.<op>; a callable must exist for the test to
    # run (LookupError otherwise).
    return flag_gems.testing.resolve_gems_op(
        "sparse_csc_tensor", getattr(flag_gems, "sparse_csc_tensor", None)
    )


def _assert_result(res_out, ref_out, dtype, index_dtype, equal_nan=False):
    """Structural and value comparison of a CSC tensor against the reference."""
    assert res_out.layout == torch.sparse_csc
    assert ref_out.layout == torch.sparse_csc
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.sparse_dim() == 2
    assert res_out.dense_dim() == 0
    assert ref_out.sparse_dim() == 2
    assert ref_out.dense_dim() == 0
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    utils.gems_assert_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    utils.gems_assert_equal(res_out.row_indices(), ref_out.row_indices())
    if dtype in _EXACT_CSC_DTYPES:
        utils.gems_assert_equal(res_out.values(), ref_out.values(), equal_nan=equal_nan)
        utils.gems_assert_equal(res_out, ref_out, equal_nan=equal_nan)
        utils.gems_assert_equal(
            res_out.to_dense(), ref_out.to_dense(), equal_nan=equal_nan
        )
    else:
        utils.gems_assert_close(
            res_out.values(), ref_out.values(), dtype, equal_nan=equal_nan
        )
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan)
        utils.gems_assert_close(
            res_out.to_dense(), ref_out.to_dense(), dtype, equal_nan=equal_nan
        )


# ---------------------------------------------------------------------------
# 2-D explicit-size constructor
# ---------------------------------------------------------------------------


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_BATCHED_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _FLOAT_CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_batched(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, seed=1, index_dtype=index_dtype, value_range=value_range
    )

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


# ---------------------------------------------------------------------------
# Shape-inferred (no-size) overload
# ---------------------------------------------------------------------------


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("ccol_list, row_list, expected_shape", _CSC_NO_SIZE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _FLOAT_CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_no_size(
    ccol_list, row_list, expected_shape, dtype, index_dtype, value_range
):
    ccol = torch.tensor(ccol_list, dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor(row_list, dtype=index_dtype, device=flag_gems.device)
    values = _make_values(len(row_list), dtype, value_range=value_range)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    assert tuple(ref_out.shape) == expected_shape
    assert tuple(res_out.shape) == tuple(ref_out.shape)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("ccol_list, row_list, expected_shape", _CSC_NO_SIZE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _EXACT_CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_no_size_exact(
    ccol_list, row_list, expected_shape, dtype, index_dtype, value_range
):
    ccol = torch.tensor(ccol_list, dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor(row_list, dtype=index_dtype, device=flag_gems.device)
    values = _make_values(len(row_list), dtype, value_range=value_range)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    assert tuple(ref_out.shape) == expected_shape
    assert tuple(res_out.shape) == tuple(ref_out.shape)


# ---------------------------------------------------------------------------
# Structural preservation: uncoalesced and unsorted entries
# ---------------------------------------------------------------------------


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_uncoalesced(dtype, index_dtype):
    # Duplicate (row, col) entries: the constructor must keep them verbatim,
    # not coalesce them (nnz stays 3 for a logical 2x2 matrix).
    ccol = torch.tensor([0, 1, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 0], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    assert torch.ops.aten._nnz(res_out) == 3
    assert torch.ops.aten._nnz(ref_out) == 3


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _CSC_DTYPES)
def test_sparse_csc_tensor_unsorted_rows(dtype, index_dtype):
    # Non-monotonic row indices within a column: legal CSC structure, kept
    # verbatim by the constructor.
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 2], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)


# ---------------------------------------------------------------------------
# Shape levels (tu.selected_shapes())
# ---------------------------------------------------------------------------


def _shape_level_cases():
    # The 0-dim scalar shape is meaningless for a 2-D sparse layout; 1-dim
    # entries become square 2-D tensors. Larger shapes are used as-is.
    cases = []
    for shape in tu.selected_shapes():
        if not shape:
            continue
        if len(shape) == 1:
            cases.append((shape + shape, shape[0]))
        else:
            cases.append((shape, min(shape[-2] * shape[-1], 8)))
    return cases


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _shape_level_cases())
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _FLOAT_CSC_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csc_tensor_shape_levels(shape, nnz, dtype, index_dtype, value_range):
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, seed=2, index_dtype=index_dtype, value_range=value_range
    )

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, index_dtype)
    assert tuple(res_out.shape) == tuple(shape)


# ---------------------------------------------------------------------------
# Boundary and non-finite values
# ---------------------------------------------------------------------------

_BOUNDARY_RANGES = [
    ["min", "min"],
    ["max", "max"],
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
]


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("value_range", _BOUNDARY_RANGES)
def test_sparse_csc_tensor_boundary_values(value_range):
    # Constant tensors at the dtype extremes must round-trip bit-exactly
    # through the construction (a pure copy).
    for dtype in _CSC_DTYPES:
        ccol, row, values = _make_csc_inputs(
            (4, 4), 4, dtype, index_dtype=torch.int64, value_range=value_range
        )

        ref_out = torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [4, 4],
            dtype=dtype,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
        gems_op = _resolve_gems_op()
        res_out = gems_op(
            ccol,
            row,
            values,
            [4, 4],
            dtype=dtype,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )

        _assert_result(res_out, ref_out, dtype, torch.int64)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("dtype", _FLOAT_CSC_DTYPES)
def test_sparse_csc_tensor_nan_inf_values(dtype):
    # Non-finite values are stored verbatim; comparison uses equal_nan=True so
    # nan == nan, inf == inf and the sign of zero are all matched.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 1.5, -0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ccol = torch.tensor([0, 2, 5], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64, device=flag_gems.device)

    ref_out = torch.ops.aten.sparse_csc_tensor(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_result(res_out, ref_out, dtype, torch.int64, equal_nan=True)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def _build_default_inputs(dtype=torch.float32, index_dtype=torch.int64):
    values = _make_values(2, dtype, value_range=["0", "1"])
    ccol = torch.tensor([0, 1, 2], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    return ccol, row, values


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_dtype_mismatch():
    # dtype kwarg contradicting the values dtype is rejected by the aten
    # reference; the candidate must raise too.
    ccol, row, values = _build_default_inputs(dtype=torch.float16)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        gems_op(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_missing_dtype():
    # Without an explicit dtype the aten op forces float32 storage and rejects
    # non-float32 values.
    ccol, row, values = _build_default_inputs(dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 1],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        gems_op(
            ccol,
            row,
            values,
            [2, 1],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_wrong_layout():
    # A non-CSC layout kwarg must be rejected.
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        gems_op(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_sparse_csc_tensor_rejects_device_mismatch():
    # Index tensors on a different device than the values are rejected.
    ccol, row, values = _build_default_inputs()
    values = values.to("cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        gems_op(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_sparse_csc_tensor_rejects_missing_device():
    # Without an explicit device kwarg the CUDA constructor cannot infer the
    # target device from the index tensors.
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        gems_op(
            ccol,
            row,
            values,
            [2, 1],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
