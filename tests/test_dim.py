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

# aten::dim(Tensor self) -> int returns the number of dimensions of a tensor,
# i.e. ``len(self.size())``, for every layout the runtime supports: strided
# (dense), sparse COO and sparse CSR. It is a pure metadata query whose result
# never depends on the stored values or the storage dtype, so every workload
# below covers a distinct (shape, layout) pair across all float, int and bool
# dtypes. The result is a plain Python int, so each workload asserts exact
# equality.
_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Dense (strided) tensors: dim == len(shape). Ranks 0 through 5 cover the full
# range, including the degenerate scalar case (rank 0).
_DENSE_CASES = [
    ((), 0),
    ((5,), 1),
    ((3, 4), 2),
    ((8, 8, 8), 3),
    ((3, 4, 2, 5), 4),
    ((3, 4, 5, 4, 5), 5),
]

# Empty dense tensors: numel == 0, but the rank is still reported exactly.
_EMPTY_DENSE_CASES = [
    ((0,), 1),
    ((0, 5), 2),
    ((2, 0, 3), 3),
]

# Sparse COO tensors: (sparse_shape, dense_shape, nnz) with logical size
# ``sparse_shape + dense_shape`` and expected result
# ``len(sparse_shape) + len(dense_shape)``. Covers all-sparse layouts as well
# as mixed sparse+dense ranks from 1 up to 5.
_COO_CASES = [
    ((4, 4), (), 8),
    ((8, 8, 8), (), 64),
    ((4, 4), (3,), 8),
    ((2, 3, 4), (5,), 12),
    ((16, 16), (7, 13), 40),
    ((2, 3, 4), (5, 6), 12),
    ((3,), (4, 5, 6), 2),
]

# Sparse CSR tensors: (shape, nnz). dim is the full logical rank for both 2-D
# and batched layouts.
_CSR_CASES = [
    ((4, 4), 3),
    ((2, 4, 4), 5),
    ((3, 5, 7), 3),
]


def _make_dense(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    # Keep the magnitude small so the values stay valid for every integer dtype.
    return torch.randint(-5, 6, shape, dtype=dtype, device=flag_gems.device)


def _make_coo(sparse_shape, dense_shape, nnz, dtype, seed=0):
    # Deterministic CPU-side generation; the sparse tensor is then constructed
    # on the test device. Duplicate indices are allowed (the layout is simply
    # uncoalesced), which is covered separately below.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values_shape = (nnz,) + tuple(dense_shape)
    if dtype.is_floating_point:
        values = torch.randn(values_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_csr(shape, nnz, dtype, seed=0):
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
    if dtype.is_floating_point:
        values = torch.randn(nnz, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (nnz,), dtype=dtype, generator=gen)
    if len(shape) == 3:
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.dim is registered; resolution order
    # is: (1) override, (2) the direct flag_gems.dim callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("dim", getattr(flag_gems, "dim", None))


def _assert_result(res_out, ref_out, expected):
    # dim returns a plain Python int holding the rank, so exact equality is
    # required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == expected


@pytest.mark.dim
@pytest.mark.parametrize("shape, expected", _DENSE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_dense(shape, expected, dtype):
    inp = _make_dense(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.dim
@pytest.mark.parametrize("shape, expected", _EMPTY_DENSE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_empty_dense(shape, expected, dtype):
    inp = _make_dense(shape, dtype)
    assert inp.numel() == 0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.dim
@pytest.mark.parametrize("case", _COO_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_sparse_coo(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


@pytest.mark.dim
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_sparse_csr(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == 0
    assert inp.sparse_dim() == 2


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_empty_coo(dtype):
    # nnz == 0: indices and values are empty, but the rank of the layout is
    # still reported exactly as for a populated tensor.
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

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dim_uncoalesced_coo(dtype):
    # The (0, 0) coordinate is repeated, so the tensor is uncoalesced; dim must
    # still report the same rank as the coalesced form because it never
    # inspects the index or data values.
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    gen = torch.Generator("cpu").manual_seed(0)
    values_shape = (5,) + tuple(dense_shape)
    if dtype.is_floating_point:
        values = torch.randn(values_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))
