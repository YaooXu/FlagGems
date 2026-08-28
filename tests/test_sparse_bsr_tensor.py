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
# (batch, nnz, br, bc) for batched (3-D) tensors, and crow/col hold the block
# grid with col entries in [0, n_col_blocks).

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
# dimension first: (batch, rows, cols). All batches share one crow/col block
# grid; the per-batch block values live in values[batch, :, :, :].
_BSR_3D_CASES = [
    ((2, 6, 6), (2, 3), [0, 2, 4, 4], [0, 1, 0, 1]),
    ((3, 4, 8), (2, 4), [0, 2, 3], [0, 1, 0]),
    ((2, 4, 4), (2, 2), [0, 2, 2], [0, 1]),
    ((2, 8, 12), (4, 3), [0, 1, 3], [0, 3, 1]),
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


def _make_index_tensor(indices):
    return torch.tensor(indices, dtype=torch.long, device=flag_gems.device)


def _make_bsr_values(nnz, block, dtype, batch=None, seed=0):
    # Deterministic CPU-side generation of the stored block values so the exact
    # structural copy semantics can be asserted; the tensor is moved to the test
    # device. Shape is (nnz, br, bc), or (batch, nnz, br, bc) for batched BSR.
    gen = torch.Generator("cpu").manual_seed(seed)
    shape = ((batch,) if batch is not None else ()) + (nnz, block[0], block[1])
    if dtype.is_floating_point:
        values = torch.randn(shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(-5, 6, shape, dtype=dtype, generator=gen)
    return values.to(flag_gems.device)


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
    else:
        assert tuple(out.values().shape) == (batch, nnz, block[0], block[1])
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
def test_sparse_bsr_tensor_crow_col_value_size(case, dtype):
    size, block, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype)
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


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_3D_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_BSR_DTYPES)
def test_sparse_bsr_tensor_crow_col_value_size_batched(case, dtype):
    size, block, crow, col = case
    batch = size[0]
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype, batch=batch)
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
def test_sparse_bsr_tensor_crow_col_value_size_exact_dtypes(case, dtype):
    # Integer and bool storages: the block values are transferred verbatim, so
    # the comparison is exact (no tolerance).
    size, block, crow, col = case
    nnz = len(col)
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype)
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
def test_sparse_bsr_tensor_crow_col_value(case, dtype):
    # Size-inferred overload: the 3-argument call (no size) derives the tensor
    # size from crow/col and the block shape of the values.
    block, crow, col = case
    nnz = len(col)
    size = ((len(crow) - 1) * block[0], (max(col) + 1) * block[1])
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype)
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
def test_sparse_bsr_tensor_crow_col_value_exact_dtypes(case, dtype):
    # Integer/bool storage through the size-inferred overload.
    block, crow, col = case
    nnz = len(col)
    size = ((len(crow) - 1) * block[0], (max(col) + 1) * block[1])
    crow_t = _make_index_tensor(crow)
    col_t = _make_index_tensor(col)
    values = _make_bsr_values(nnz, block, dtype)
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
