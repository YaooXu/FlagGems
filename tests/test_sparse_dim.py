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

# aten::sparse_dim(Tensor self) -> int returns the number of sparse dimensions
# of a tensor for every layout the runtime supports: strided (dense) tensors
# always report 0, sparse COO tensors report the number of leading sparse dims
# (``len(sparse_shape)``), and sparse CSR tensors report 2 for both 2-D and
# batched layouts. It is a pure metadata query whose result never depends on
# the stored values or the storage dtype, so every workload below covers a
# distinct (shape, layout) pair. The result is a plain Python int, so each
# workload asserts exact equality.
#
# Coverage:
#   * layouts: strided tensors (ranks 0-8 plus empty), sparse COO (all-sparse
#     and hybrid), and sparse CSR (2-D and batched 3-D), selected by --quick;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the reported sparse-dim count is identical for
#     all of them);
#   * edge cases: empty dense/COO/CSR tensors, uncoalesced COO, and
#     nan/inf/-inf/±0.0 stored values;
#   * negative: non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).
_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

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

# Higher-rank strided tensors for the "all" level (default, no --quick).
_DENSE_CASES_ALL = [
    ((3, 6, 4, 4, 6, 5, 4), 0),
    ((7, 3, 12, 4, 2, 15, 2, 2), 0),
]

# Empty dense tensors: numel == 0, but the rank is still reported and the
# number of sparse dims remains 0.
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

# Higher-rank hybrid layouts for the "all" level (default, no --quick).
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

# Additional batched CSR layout for the "all" level (default, no --quick).
_CSR_CASES_ALL = [
    ((3, 4, 4), 4),
]

# Empty CSR tensors: nnz == 0, plain 2-D and batched 3-D layouts.
_EMPTY_CSR_CASES = [
    (4, 4),
    (3, 4, 4),
]


def _dense_cases():
    """(shape, expected) strided layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 0)]
    if tu.LEVEL == "all":
        return _DENSE_CASES_CORE + _DENSE_CASES_ALL


def _coo_cases():
    """(sparse_shape, dense_shape, nnz) COO layouts selected by --quick."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    if tu.LEVEL == "all":
        return _COO_CASES_CORE + _COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid COO layouts for the range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    if tu.LEVEL == "all":
        return [((3, 4), (), 7), ((3, 4), (3,), 8), ((12, 9, 3, 6), (4,), 9)]


def _csr_cases():
    """(shape, nnz) CSR layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    if tu.LEVEL == "all":
        return _CSR_CASES_CORE + _CSR_CASES_ALL


def _csr_value_range_cases():
    """Representative 2-D + batched CSR layouts for the range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    if tu.LEVEL == "all":
        return [((4, 4), 3), ((2, 4, 4), 5)]


def _make_dense(shape, dtype, value_range):
    # Values come from the shared value-range helper; the reported sparse-dim
    # count never depends on them.
    return tu.make_input(dtype, shape, value_range)


def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper (tu.make_input) and the sparse tensor is created
    # on the test device. Duplicate indices are allowed (the layout is simply
    # uncoalesced), which is covered explicitly below.
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
    values = tu.make_input(dtype, (nnz,), value_range)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the layout stays 2-D sparse for every batch.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
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


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape, expected", _dense_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense(shape, expected, dtype):
    inp = _make_dense(shape, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape, expected", _EMPTY_DENSE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_empty_dense(shape, expected, dtype):
    inp = _make_dense(shape, dtype, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_dense_value_ranges(shape, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the reported sparse-dim count never changes
    # because sparse_dim reads only layout metadata.
    inp = _make_dense(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 0)


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_sparse_coo(case, dtype):
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
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_sparse_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape))


@pytest.mark.sparse_dim
@pytest.mark.parametrize("case", _csr_cases())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_sparse_csr(case, dtype):
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
@pytest.mark.parametrize("case", _csr_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_dim_sparse_csr_value_ranges(case, value_range, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sparse_dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


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


@pytest.mark.sparse_dim
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
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
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
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
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
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


@pytest.mark.sparse_dim
def test_sparse_dim_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises. The candidate must fail too
    # rather than silently report a bogus sparse-dim count.
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
