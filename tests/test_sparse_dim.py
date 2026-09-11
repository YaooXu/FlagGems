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

# aten::sparse_dim(Tensor self) -> int returns the number of *sparse*
# dimensions of a tensor for every layout the runtime supports: strided
# (dense) tensors always report 0, sparse COO tensors report the number of
# leading sparse dims (``len(sparse_shape)``) and sparse CSR tensors report 2
# for both 2-D and batched layouts (and for CSR tensors carrying dense dims).
# It is a pure metadata query whose result never depends on the stored values
# or on the storage dtype, so every workload below covers a distinct
# (shape, layout) pair. The result is a plain Python int, so each workload
# asserts exact equality.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * dtypes -- the probe-selected set of the required dtypes
#     (int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/bf16/fp16/int32/int64) plus
#     fp64/int16/bool where the device supports them;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) crossed with the tu.selected_shapes() levels, both for dense
#     tensors (sparse_dim == 0) and for all-sparse COO / hybrid COO / CSR
#     layouts;
#   * layouts -- strided tensors (ranks 0-8 plus empty), sparse COO
#     (all-sparse and hybrid) and sparse CSR (2-D, batched 3-D and CSR with
#     dense dims), selected by the quick / full level;
#   * edge cases -- empty dense/COO/CSR tensors, uncoalesced COO, CSR with
#     dense dims, and nan/inf/-inf/±0.0 stored values;
#   * negative cases -- non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).

# ---------------------------------------------------------------------------
# Dtype probing (device-portable: never turn an unsupported op/dtype pair into
# a red test)
# ---------------------------------------------------------------------------
_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)


def _sparse_dtype_probe(op_name, dtype):
    """Report whether ``op_name`` accepts a tiny sparse COO tensor of ``dtype``."""
    del op_name
    try:
        indices = torch.zeros(2, 1, dtype=torch.long, device=flag_gems.device)
        values = torch.zeros(1, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_coo_tensor(indices, values, (1, 1), device=flag_gems.device)
        return isinstance(torch.ops.aten.sparse_dim(utils.to_reference(inp)), int)
    except Exception:
        return False


# Probe the device before parametrizing: an op/dtype pair that cannot run must
# not be turned into a red test. If the probe yields nothing, keep the full
# candidate list rather than a float32-only fallback, so a failed/absent probe
# never silently drops the spec-required int8/uint8/fp8 dtypes.
_DTYPES = tu.supported_dtypes(
    "sparse_dim", candidates=_DTYPE_CANDIDATES, probe=_sparse_dtype_probe
) or list(_DTYPE_CANDIDATES)
_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# ---------------------------------------------------------------------------
# Layout cases
# ---------------------------------------------------------------------------
# Dense (strided) tensors: there are no sparse dims, so sparse_dim == 0 for
# every rank, including the degenerate scalar case (rank 0).
_DENSE_CASES_CORE = [
    ((), 0),
    ((5,), 0),
    ((3, 4), 0),
    ((8, 8, 8), 0),
    ((3, 4, 2, 5), 0),
    ((3, 4, 5, 4, 5), 0),
]

# Higher-rank strided tensors for the full level (default, no --quick).
_DENSE_CASES_ALL = [
    ((3, 6, 4, 4, 6, 5, 4), 0),
    ((7, 3, 12, 4, 2, 15, 2, 2), 0),
]

# Empty dense tensors: numel == 0, but the number of sparse dims is still 0.
_EMPTY_DENSE_CASES = [
    ((0,), 0),
    ((0, 5), 0),
    ((2, 0, 3), 0),
]

# Sparse COO tensors: (sparse_shape, dense_shape, nnz) with logical size
# ``sparse_shape + dense_shape`` and expected result ``len(sparse_shape)``.
# Covers all-sparse layouts as well as mixed sparse+dense ranks from 1 up to 5.
_COO_CASES_CORE = [
    ((4, 4), (), 8),
    ((8, 8, 8), (), 64),
    ((4, 4), (3,), 8),
    ((2, 3, 4), (5,), 12),
    ((16, 16), (7, 13), 40),
    ((2, 3, 4), (5, 6), 12),
    ((3,), (4, 5, 6), 2),
]

# Higher-rank hybrid layouts for the full level (default, no --quick).
_COO_CASES_ALL = [
    ((12, 9, 3, 6), (4,), 9),
    ((3, 4, 2, 5, 3), (4, 2), 11),
]

# Sparse CSR tensors: (shape, nnz). The compressed sparse layout is always 2-D
# sparse, so sparse_dim == 2 for both plain and batched layouts.
_CSR_CASES_CORE = [
    ((4, 4), 3),
    ((2, 4, 4), 5),
    ((3, 5, 7), 3),
]

# Additional batched CSR layout for the full level (default, no --quick).
_CSR_CASES_ALL = [
    ((3, 4, 4), 4),
]

# Empty CSR tensors: nnz == 0, plain 2-D and batched 3-D layouts.
_EMPTY_CSR_CASES = [
    (4, 4),
    (3, 4, 4),
]

# Fixed stored-entry count for the spec-shape COO sweeps: small so that every
# rank stays cheap, and > 1 so duplicate (uncoalesced) coordinates are
# exercised for small index spaces.
_SPEC_NNZ = 6


def _dense_cases():
    """(shape, expected) strided layouts selected by quick / full level."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 0)]
    return _DENSE_CASES_CORE + _DENSE_CASES_ALL


def _coo_cases():
    """(sparse_shape, dense_shape, nnz) COO layouts selected by quick / full."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    return _COO_CASES_CORE + _COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid COO layouts for the range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    return [((3, 4), (), 7), ((3, 4), (3,), 8), ((12, 9, 3, 6), (4,), 9)]


def _csr_cases():
    """(shape, nnz) CSR layouts selected by quick / full level."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    return _CSR_CASES_CORE + _CSR_CASES_ALL


def _spec_shapes(min_rank=0, max_rank=None):
    """``tu.selected_shapes()`` filtered to the ranks a layout supports."""
    shapes = [shape for shape in tu.selected_shapes() if len(shape) >= min_rank]
    if max_rank is not None:
        shapes = [shape for shape in shapes if len(shape) <= max_rank]
    return shapes


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper and the sparse tensor is created on the test
    # device. Duplicate indices are allowed (the layout is simply uncoalesced),
    # which is covered explicitly below.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_coo_all_sparse(shape, nnz, dtype, value_range, seed=0):
    """Map a dense spec shape onto an all-sparse COO layout of the same rank."""
    return _make_coo(tuple(shape), (), nnz, dtype, value_range, seed=seed)


def _make_csr(shape, nnz, dtype, value_range, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    col_indices = torch.randint(0, cols, (nnz,), dtype=torch.long, generator=gen)
    cuts = torch.sort(
        torch.randint(0, nnz + 1, (rows - 1,), dtype=torch.long, generator=gen)
    ).values
    crow_indices = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            cuts,
            torch.full((1,), nnz, dtype=torch.long),
        ]
    )
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the layout stays 2-D sparse for every batch.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = tu.make_input(dtype, (shape[0], nnz), value_range)
    else:
        values = tu.make_input(dtype, (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_csr_with_dense_dims(dtype, value_range):
    """CSR layout carrying a dense dim: shape (rows, cols, dense)."""
    rows, cols, dense, nnz = 4, 4, 3, 5
    # crow segments: row0 -> 1, row1 -> 1, row2 -> 2, row3 -> 1 stored block.
    crow = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long, device=flag_gems.device)
    col = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(dtype, (nnz, dense), value_range)
    return torch.sparse_csr_tensor(
        crow, col, values, (rows, cols, dense), device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build a CSR tensor with nnz == 0: the layout is still 2-D sparse, so
    sparse_dim is reported exactly as for a populated tensor."""
    if len(shape) == 2:
        rows, _ = shape
        crow_indices = torch.zeros(rows + 1, dtype=torch.long, device=flag_gems.device)
        col_indices = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        _, rows, _ = shape
        crow_indices = torch.zeros(
            shape[0], rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_indices = torch.empty(
            shape[0], 0, dtype=torch.long, device=flag_gems.device
        )
        values = torch.empty(shape[0], 0, dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_dim is registered; resolution
    # order is: (1) override, (2) the direct flag_gems.sparse_dim callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_dim", getattr(flag_gems, "sparse_dim", None)
    )


def _assert_result(res_out, ref_out, expected):
    # sparse_dim returns a plain Python int holding the number of sparse dims,
    # so exact equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == expected


# ---------------------------------------------------------------------------
# Dense (strided) layouts: sparse_dim == 0
# ---------------------------------------------------------------------------
@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape, expected", _dense_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense_layouts(shape, expected, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape, expected", _EMPTY_DENSE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_dense(shape, expected, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _spec_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense_spec_shapes_value_ranges(shape, value_range, dtype):
    # Shape level from the shared spec set (0-dim .. 5-dim) crossed with the
    # five spec value ranges on dense tensors. The stored values never change
    # the result (always 0 sparse dims), but they exercise the value-range
    # machinery end to end.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 0)


# ---------------------------------------------------------------------------
# Sparse COO layouts: sparse_dim == len(sparse_shape)
# ---------------------------------------------------------------------------
@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape))
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _spec_shapes(min_rank=1))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_spec_shapes_value_ranges(shape, value_range, dtype):
    # Shared spec shape set mapped onto all-sparse COO (sparse_dim == rank)
    # crossed with the five spec value ranges.
    inp = _make_coo_all_sparse(shape, _SPEC_NNZ, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape))


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_coo(dtype):
    # nnz == 0: indices and values are empty, but the number of sparse dims of
    # the layout is still reported exactly as for a populated tensor.
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape))


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_uncoalesced_coo(dtype):
    # The (0, 0) coordinate is repeated, so the tensor is uncoalesced;
    # sparse_dim must still report the same value as the coalesced form because
    # it never inspects the index or data values.
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = tu.make_input(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape))


# ---------------------------------------------------------------------------
# Sparse CSR layouts: sparse_dim == 2
# ---------------------------------------------------------------------------
@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _csr_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_layouts(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 0


@pytest.mark.sparse_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_value_ranges(value_range, dtype):
    # The CSR path is value-independent too: sweep the spec ranges through a
    # 2-D CSR tensor with a fixed 5-entry crow/col pattern.
    shape, nnz = (4, 4), 5
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_dense_dims(value_range, dtype):
    # CSR layout with a dense dimension: the compressed layout is still 2-D
    # sparse, independent of the trailing dense block.
    inp = _make_csr_with_dense_dims(dtype, value_range)
    assert inp.sparse_dim() == 2
    assert inp.dense_dim() == 1
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _EMPTY_CSR_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_csr(shape, dtype):
    # nnz == 0: indices and values are empty, but the compressed layout is
    # still 2-D sparse, so sparse_dim is reported exactly as for a populated
    # tensor.
    inp = _make_empty_csr(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", _spec_shapes(min_rank=2, max_rank=3))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_csr_spec_shapes(shape, dtype):
    # Shared spec shape set restricted to the ranks CSR can represent (2-D and
    # batched 3-D).
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


# ---------------------------------------------------------------------------
# nan/inf payload
# ---------------------------------------------------------------------------
@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_sparse_dim_nan_inf_dense(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values for a metadata query: the
    # strided path still reports 0 sparse dims.
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    ).reshape(2, 3)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 0)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_sparse_dim_nan_inf_coo(dtype):
    # The same values stored sparsely: sparse_dim reports the number of sparse
    # dims of the layout (1 for this 1-D layout) regardless of the nan/inf
    # payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 1)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_sparse_dim_nan_inf_csr(dtype):
    # The same values stored in CSR form: sparse_dim reports 2 for the
    # compressed 2-D sparse layout regardless of the nan/inf payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------
@pytest.mark.sparse_dim
def test_sparse_dim_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises. The candidate must fail too
    # rather than silently report a bogus sparse-dim count.
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_dim(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AttributeError)
    ):
        _resolve_gems_op()(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AttributeError)
    ):
        _resolve_gems_op()(None)
