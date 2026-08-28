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

# aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices,
#     Tensor col_indices, Tensor values, int[] size, *,
#     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=False) -> Tensor constructs a sparse CSR tensor of the
# given ``size`` from raw index tensors and values.
#
# aten::sparse_csr_tensor.crow_col_value(Tensor crow_indices,
#     Tensor col_indices, Tensor values, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False)
#     -> Tensor is the size-inferred variant: rows = len(crow)-1,
#     cols = max(col)+1.
#
# The two overloads share one public name, and ``torch.ops.aten.sparse_csr_tensor``
# dispatches between them by argument count (4 args -> size variant, 3 args ->
# inferred variant); the candidate under test is the same public callable, so
# every reference call below mirrors the candidate call exactly. The ``dtype``
# keyword is always passed explicitly: without it the aten op forces the values
# to float32 and raises RuntimeError for any other storage dtype. The ``device``
# keyword is passed explicitly too: on CUDA this torch build fails to infer the
# device from the input tensors ("Values and compressed tensor instance need to
# be on the same device") unless the target device is given.
#
# CSR layout facts exercised below: layout == torch.sparse_csr, sparse_dim == 2,
# dense_dim == 0 (batched CSR carries the batch dims separately), crow/col are
# the (compressed, plain) index arrays with col entries in [0, n_cols), and
# values are stored verbatim (the factory performs no arithmetic on them).

# Each 2-D case is (size, crow_indices, col_indices): a square matrix with
# empty trailing rows, ragged rows with an empty middle row, non-square
# matrices, single-row/single-column extremes, and a dense first row.
_CSR_2D_CASES = [
    ((4, 4), [0, 2, 4, 4, 4], [0, 1, 0, 1]),
    ((5, 4), [0, 2, 3, 3, 5, 5], [0, 1, 2, 0, 3]),
    ((3, 6), [0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ((1, 4), [0, 2], [0, 3]),
    ((6, 1), [0, 1, 1, 2, 2, 3, 3], [0, 0, 0]),
    ((7, 5), [0, 1, 3, 4, 4, 6, 6, 6], [0, 4, 2, 1, 3]),
    ((3, 3), [0, 3, 4, 6], [0, 1, 2, 0, 1, 2]),
]

# Each batched case is (size, crow_indices, col_indices) with the batch
# dimension first: (batch, rows, cols). Every batch stores the same nnz; the
# crow/col index arrays have shape (batch, rows+1) / (batch, nnz) and the
# per-batch values live in values[batch, :].
_CSR_3D_CASES = [
    ((2, 3, 4), [[0, 2, 4, 5], [0, 1, 3, 4]], [[0, 1, 0, 2], [1, 0, 2, 3]]),
    (
        (3, 4, 5),
        [[0, 1, 3, 4, 5], [0, 2, 2, 3, 5], [0, 1, 2, 4, 5]],
        [[0, 3, 1, 2, 4], [4, 0, 2, 3, 1], [1, 3, 0, 4, 2]],
    ),
    (
        (2, 5, 4),
        [[0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]],
        [[0, 1, 3, 0, 2, 1], [3, 2, 0, 1, 3, 0]],
    ),
]

# Empty storage (nnz == 0): the row-pointer array still has its full
# rows+1 entries but stores no column/values. batch is None for the 2-D case.
_CSR_EMPTY_CASES = [
    ((4, 5), None),
    ((2, 4, 5), 2),
]

# Size-inferred ``crow_col_value`` cases as (crow_indices, col_indices).
# Expected size is derived from crow/col: rows = len(crow)-1,
# cols = max(col)+1. The inference for batched inputs is underspecified, so
# only 2-D values are covered (the same convention as the BSR factory tests).
_CSR_2D_INFERRED_CASES = [
    ([0, 2, 4], [0, 1, 0, 1]),
    ([0, 2, 2, 4], [0, 1, 0, 1]),
    ([0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ([0, 1, 2], [0, 0]),
]

# The op copies the given values verbatim into the new tensor (no arithmetic),
# so every float storage dtype the runtime supports is fair game, and exact
# equality holds for integer/bool storages.
_FLOAT_CSR_DTYPES = utils.ALL_FLOAT_DTYPES
_EXACT_CSR_DTYPES = utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_index_tensor(indices, index_dtype=torch.long):
    return torch.tensor(indices, dtype=index_dtype, device=flag_gems.device)


def _make_csr_values(nnz, dtype, batch=None, seed=0):
    # Deterministic CPU-side generation of the stored values so the exact
    # structural copy semantics can be asserted; the tensor is moved to the
    # test device. Shape is (nnz,), or (batch, nnz) for batched CSR.
    gen = torch.Generator("cpu").manual_seed(seed)
    shape = ((batch,) if batch is not None else ()) + (nnz,)
    if dtype.is_floating_point:
        values = torch.randn(shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(-5, 6, shape, dtype=dtype, generator=gen)
    return values.to(flag_gems.device)


def _assert_csr_structure(out, size, nnz, dtype, batch=None, crow_len=None):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # sparse/dense split, index-array sizes, and the nnz count.
    assert out.layout == torch.sparse_csr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == 0
    assert out._nnz() == nnz
    expected_crow_len = size[-2] + 1 if crow_len is None else crow_len
    if batch is None:
        assert tuple(out.values().shape) == (nnz,)
        assert len(out.crow_indices()) == expected_crow_len
        assert len(out.col_indices()) == nnz
    else:
        assert tuple(out.values().shape) == (batch, nnz)
        assert tuple(out.crow_indices().shape) == (batch, expected_crow_len)
        assert tuple(out.col_indices().shape) == (batch, nnz)
    assert (out.col_indices() >= 0).all()
    assert (out.col_indices() < size[-1]).all()


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_csr_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_csr_tensor", getattr(flag_gems, "sparse_csr_tensor", None)
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size(case, dtype):
    size, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    # Values are stored verbatim, so the float comparison is exact within
    # tolerance; the index arrays must match bit-for-bit.
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES)
@pytest.mark.parametrize("dtype", _EXACT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_exact_dtypes(case, dtype):
    # Integer and bool storages: the values are transferred verbatim, so the
    # comparison is exact (no tolerance).
    size, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_3D_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_batched(case, dtype):
    size, crow, col = case
    batch = size[0]
    nnz = len(col[0])
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype, batch=batch)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype, batch=batch)
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_empty(case, dtype):
    size, batch = case
    n_rows = size[-2]
    if batch is None:
        crow_t = torch.zeros(n_rows + 1, dtype=torch.long, device=flag_gems.device)
        col_t = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(0, dtype)
    else:
        crow_t = torch.zeros(
            batch, n_rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_t = torch.empty(batch, 0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(0, dtype, batch=batch)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, 0, dtype, batch=batch)
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES[:3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_index_dtypes(case, index_dtype, dtype):
    # The index arrays are accepted as either int32 or int64 and the resulting
    # sparse tensor keeps that index dtype.
    size, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow, index_dtype=index_dtype)
    col_t = _make_index_tensor(col, index_dtype=index_dtype)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    assert res_out.crow_indices().dtype == index_dtype
    assert res_out.col_indices().dtype == index_dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_trailing_empty_rows(dtype):
    # The explicit size may claim more rows than the crow_indices describe; the
    # extra trailing rows are empty and the stored crow_indices keep their
    # original (shorter) length. The reference comparison is authoritative here
    # because the whole-tensor assert_close covers the index arrays too.
    size = (5, 2)
    crow = [0, 2, 4]
    col = [0, 1, 0, 1]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, list(size), dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype, crow_len=len(crow))
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value(case, dtype):
    # Size-inferred overload: the 3-argument call (no size) derives the tensor
    # size from crow/col: rows = len(crow)-1, cols = max(col)+1.
    crow, col = case
    nnz = len(col)
    size = (len(crow) - 1, max(col) + 1)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _EXACT_CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_exact_dtypes(case, dtype):
    # Integer/bool storage through the size-inferred overload.
    crow, col = case
    nnz = len(col)
    size = (len(crow) - 1, max(col) + 1)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_csr_values(nnz, dtype)
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = _resolve_gems_op()(
        crow_t, col_t, values, dtype=dtype, device=crow_t.device
    )

    _assert_csr_structure(res_out, size, nnz, dtype)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values())
