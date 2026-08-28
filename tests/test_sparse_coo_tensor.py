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
from . import conftest as cfg  # noqa: E402

# aten::sparse_coo_tensor(Tensor indices, Tensor values, *, ScalarType? dtype,
#     Layout? layout, Device? device, bool? pin_memory) -> Tensor (the
#     size-inferred ``indices`` overload) and
# aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ...)
#     -> Tensor (the explicit ``indices_size`` overload) construct a COO sparse
#     tensor from raw index/values components, and
# aten::sparse_coo_tensor(int[] size, *, ScalarType? dtype, ...) -> Tensor
#     (the ``size`` overload) returns the empty sparse tensor of that shape.
#
# The three overloads share one public name and dispatch by argument count /
# shape (3 args -> size-only, 2 args -> inferred, 2 args + size -> explicit),
# so the candidate under test is the same public callable and every reference
# call below mirrors the candidate call exactly. The ``dtype`` keyword is
# always passed explicitly (without it the aten op forces float32), and
# ``device`` is passed explicitly as well so the CUDA reference does not have
# to infer the device from the component tensors.
#
# COO layout facts exercised below: layout == torch.sparse_coo, indices shape
# is (sparse_dim, nnz) with dtype int64, values shape is (nnz,) + dense_shape
# where dense_shape == size[sparse_dim:], sparse_dim + dense_dim == ndim.
# The constructor stores the raw components verbatim: duplicate / unsorted
# indices stay uncoalesced (is_coalesced() == False) exactly like the
# reference, while nnz == 0 inputs are coalesced. An explicit
# ``is_coalesced=True`` kwarg is honored by the reference and asserted on the
# result.

# Each 2-D case is (size, indices): indices is a Python list with one inner
# list per sparse dimension and nnz columns. The column order deliberately
# repeats / reorders coordinates so the constructed tensor is uncoalesced,
# exercising verbatim storage of the raw components.
_COO_2D_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((6, 8), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((5, 6), [[2, 4, 1], [3, 0, 5]]),
    ((3, 3), [[0, 2, 1, 2], [0, 1, 2, 0]]),
]

# Multi-dim cases: (size, indices) where sparse_dim == len(indices) and the
# trailing dims of size are dense (values shape (nnz,) + size[sparse_dim:]).
# Covers dense dims (2 sparse + 1/2 dense) and 3 sparse dims.
_COO_ND_CASES = [
    ((3, 4, 5), [[0, 1, 2, 1], [1, 3, 0, 2]]),
    ((2, 3, 7), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2], [1, 3, 0]]),
    ((3, 3, 3), [[0, 2, 1], [1, 0, 2], [2, 1, 0]]),
]

# Size-inferred cases: (expected_size, indices, dense_shape). The sparse part
# of the size is max(index[d]) + 1 per sparse dim, the dense part comes from
# the values shape (nnz,) + dense_shape. The last case has a dense dim.
_COO_INFERRED_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]], ()),
    ((5, 6), [[2, 4, 1], [3, 0, 5]], ()),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# Sizes for the size-only overload: the empty sparse tensor (nnz == 0) with
# sparse_dim == len(size) and dense_dim == 0.
_COO_SIZE_ONLY_CASES = [
    ((5,),),
    ((2, 3),),
    ((4, 5, 6),),
]

# Explicit-size empty cases: (size, sparse_dim). nnz == 0 but the block grid
# shape and dense dims are still carried by the tensor.
_COO_EMPTY_CASES = [
    ((2, 3), 2),
    ((4, 5, 6), 2),
    ((2, 3, 4, 5), 2),
    ((3, 4, 5), 3),
]

# The factory accepts every storage dtype the sparse COO runtime supports; the
# float comparison uses the usual tolerance policy while integer/bool storages
# are compared exactly.
_FLOAT_COO_DTYPES = utils.ALL_FLOAT_DTYPES
_EXACT_COO_DTYPES = utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_index_tensor(indices):
    return torch.tensor(indices, dtype=torch.long, device=flag_gems.device)


def _make_values(nnz, dense_shape, dtype, seed=0):
    # Deterministic CPU-side generation of the stored values so the verbatim
    # copy semantics can be asserted; the tensor is moved to the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    shape = (nnz,) + tuple(dense_shape)
    if dtype.is_floating_point:
        values = torch.randn(shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(-5, 6, shape, dtype=dtype, generator=gen)
    return values.to(flag_gems.device)


def _assert_coo_structure(
    res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim, is_coalesced=None
):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # device, sparse/dense split, the nnz count and the coalesced flag.
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert tuple(res_out.shape) == tuple(size)
    assert tuple(ref_out.shape) == tuple(size)
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == dense_dim
    assert ref_out.sparse_dim() == sparse_dim
    assert ref_out.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(res_out) == nnz
    assert torch.ops.aten._nnz(ref_out) == nnz
    assert tuple(torch.ops.aten._indices(res_out).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._values(res_out).shape) == (nnz,) + tuple(
        size[sparse_dim:]
    )
    # The constructor records the coalesced flag verbatim, so the candidate
    # must match the reference exactly (uncoalesced for nnz > 0, coalesced for
    # nnz == 0 unless is_coalesced=True is passed).
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    if is_coalesced is not None:
        assert res_out.is_coalesced() == is_coalesced
        assert ref_out.is_coalesced() == is_coalesced
    # The index tensors are exact integer data stored verbatim.
    utils.gems_assert_equal(
        torch.ops.aten._indices(res_out), torch.ops.aten._indices(ref_out)
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_coo_tensor is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_coo_tensor", getattr(flag_gems, "sparse_coo_tensor", None)
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_size(case, dtype):
    # Size-only overload: an empty sparse tensor (nnz == 0, coalesced) with the
    # requested shape and storage dtype.
    (size,) = case
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten.sparse_coo_tensor(
        list(size), dtype=dtype, device=ref_device
    )
    res_out = _resolve_gems_op()(list(size), dtype=dtype, device=flag_gems.device)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, len(size), 0)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _EXACT_COO_DTYPES)
def test_sparse_coo_tensor_size_exact_dtypes(case, dtype):
    # Integer/bool storages through the size-only overload: exact comparison.
    (size,) = case
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten.sparse_coo_tensor(
        list(size), dtype=dtype, device=ref_device
    )
    res_out = _resolve_gems_op()(list(size), dtype=dtype, device=flag_gems.device)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, len(size), 0)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_2D_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size(case, dtype):
    # Explicit-size overload (2 sparse dims, no dense dims). The components are
    # stored verbatim, so duplicate / unsorted coordinates stay uncoalesced.
    size, indices = case
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, (), dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, list(size), dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_2D_CASES)
@pytest.mark.parametrize("dtype", _EXACT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_exact_dtypes(case, dtype):
    # Integer/bool storages through the explicit-size overload.
    size, indices = case
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, (), dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, list(size), dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(
        torch.ops.aten._values(res_out), torch.ops.aten._values(ref_out)
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_ND_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_nd(case, dtype):
    # Multi-dim overload coverage: dense dims (sparse_dim=2) and 3 sparse dims.
    size, indices = case
    sparse_dim = len(indices)
    dense_dim = len(size) - sparse_dim
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, tuple(size[sparse_dim:]), dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, list(size), dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices(case, dtype):
    # Size-inferred overload: the sparse part of the size is max(index[d]) + 1
    # per sparse dim and the dense part comes from the values shape.
    size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_empty(case, dtype):
    # Explicit-size overload with nnz == 0: the index/values storage is empty
    # but the requested shape, dense dims and dtype are carried by the tensor.
    size, sparse_dim = case
    dense_shape = tuple(size[sparse_dim:])
    indices_t = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, dense_shape, dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, list(size), dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(
        res_out, ref_out, size, 0, dtype, sparse_dim, len(dense_shape)
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _EXACT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_empty_exact_dtypes(case, dtype):
    # Integer/bool storages with nnz == 0.
    size, sparse_dim = case
    dense_shape = tuple(size[sparse_dim:])
    indices_t = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, dense_shape, dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )
    res_out = _resolve_gems_op()(
        indices_t, values, list(size), dtype=dtype, device=indices_t.device
    )

    _assert_coo_structure(
        res_out, ref_out, size, 0, dtype, sparse_dim, len(dense_shape)
    )
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_is_coalesced(dtype):
    # An explicit is_coalesced=True kwarg is honored by the constructor even
    # when the stored coordinates contain duplicates (the flag is recorded
    # verbatim); the result must be coalesced like the reference.
    size = (2, 3)
    indices = [[0, 1, 1], [2, 0, 2]]
    nnz = 3
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, (), dtype)
    ref_indices = utils.to_reference(indices_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_indices.device,
        is_coalesced=True,
    )
    res_out = _resolve_gems_op()(
        indices_t,
        values,
        list(size),
        dtype=dtype,
        device=indices_t.device,
        is_coalesced=True,
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0, is_coalesced=True)
    utils.gems_assert_close(res_out, ref_out, dtype)
