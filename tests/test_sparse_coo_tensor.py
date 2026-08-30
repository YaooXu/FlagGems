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
from . import test_utils as tu  # noqa: E402

# aten::sparse_coo_tensor(Tensor indices, Tensor values, *, ScalarType? dtype,
#     Layout? layout, Device? device, bool? pin_memory) -> Tensor (the
#     size-inferred ``indices`` overload),
# aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ...)
#     -> Tensor (the explicit ``indices_size`` overload) and
# aten::sparse_coo_tensor(int[] size, *, ScalarType? dtype, ...) -> Tensor
#     (the ``size`` overload) construct a sparse COO tensor. The three overloads
#     share one public name and dispatch by argument count / shape (1 arg ->
#     size-only, 2 args -> inferred, 2 args + size -> explicit), so the
#     candidate under test is the same public callable and every reference call
#     below mirrors the candidate call exactly. The ``dtype`` keyword is always
#     passed explicitly (without it the aten op forces float32), and ``device``
#     is passed explicitly as well so the CUDA reference does not have to infer
#     the device from the component tensors.
#
# COO layout facts exercised below: layout == torch.sparse_coo, indices shape
# is (sparse_dim, nnz) with dtype int64, values shape is (nnz,) + dense_shape
# where dense_shape == size[sparse_dim:], sparse_dim + dense_dim == ndim.
# The constructor stores the raw components verbatim: duplicate / unsorted
# indices stay uncoalesced (is_coalesced() == False) exactly like the
# reference, while nnz == 0 inputs are coalesced. An explicit
# ``is_coalesced=True`` kwarg is honored by the reference and asserted on the
# result.
#
# The op is a pure sparse factory: it performs no arithmetic on the stored
# values (nan/inf/-0.0 and dtype-extreme values survive unchanged) and it is
# neither differentiable nor broadcastable, so the backward and broadcast
# dimensions of the regular-operator spec do not apply; the value-range, shape,
# nan/inf and negative dimensions are covered here instead. Shape levels follow
# the sparse-COO grid: 0-dim scalars and 5+ dim dense shapes do not map onto
# sparse COO indexing, so dedicated (size, indices) cases replace the dense
# ``tu.selected_shapes()`` set (same approach as test_coalesce.py).

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
# Covers 1 sparse dim, dense dims (2 sparse + 1/2 dense) and 3 sparse dims.
_COO_ND_CASES = [
    ((5,), [[0, 2, 4]]),
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

# Value-range sweep subset: (variant, size, indices, dense_shape) covering the
# explicit-size 2-D path, the explicit-size ND path with dense dims and the
# size-inferred path. For the inferred variant ``size`` is the expected result
# shape (the sparse dims are max(index[d]) + 1 and the dense dims come from the
# values shape).
_COO_VALUE_CASES = [
    ("indices_size", (2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ("indices_size", (2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]], (4, 5)),
    ("indices", (2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# The factory accepts every storage dtype the sparse COO runtime supports; the
# float comparison uses the usual tolerance policy while integer/bool storages
# are compared exactly.
_FLOAT_COO_DTYPES = utils.ALL_FLOAT_DTYPES
_EXACT_COO_DTYPES = utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Integer/bool value ranges: one negative, one positive plus the dtype
# extremes. The full selected_ranges() sweep is reserved for floats (the
# non-extreme float ranges are meaningless for the exact integer path).
_INT_VALUE_RANGES = [
    ["-1", "1"],
    ["min", "0"],
    ["0", "max"],
]


def _make_index_tensor(indices):
    return torch.tensor(indices, dtype=torch.long, device=flag_gems.device)


def _make_values(nnz, dense_shape, dtype, value_range=None):
    # Stored values come from the shared value-range framework (tu.make_input):
    # range-bound symbols resolve per-dtype, so every storage dtype gets valid
    # inputs within the requested numeric range. Construction copies the raw
    # entries verbatim, so any representable value round-trips exactly.
    if value_range is None:
        value_range = ["-1", "1"]
    shape = (nnz,) + tuple(dense_shape)
    return tu.make_input(dtype, shape, value_range).to(flag_gems.device)


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


def _call_reference(indices, values, size, dtype):
    # Mirrors the candidate call exactly. size=None selects the size-inferred
    # ``indices`` overload, a list selects the explicit ``indices_size``
    # overload; the component tensors are moved to the reference device.
    ref_indices = utils.to_reference(indices)
    ref_values = utils.to_reference(values)
    if size is None:
        return torch.ops.aten.sparse_coo_tensor(
            ref_indices, ref_values, dtype=dtype, device=ref_indices.device
        )
    return torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )


def _call_candidate(indices, values, size, dtype):
    if size is None:
        return _resolve_gems_op()(indices, values, dtype=dtype, device=indices.device)
    return _resolve_gems_op()(
        indices, values, list(size), dtype=dtype, device=indices.device
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

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

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

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(
        torch.ops.aten._values(res_out), torch.ops.aten._values(ref_out)
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_ND_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_nd(case, dtype):
    # Multi-dim overload coverage: 1 sparse dim, dense dims (sparse_dim=2) and
    # 3 sparse dims.
    size, indices = case
    sparse_dim = len(indices)
    dense_dim = len(size) - sparse_dim
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, tuple(size[sparse_dim:]), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

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

    ref_out = _call_reference(indices_t, values, None, dtype)
    res_out = _call_candidate(indices_t, values, None, dtype)

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

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

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

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

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


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_float_value_ranges(case, value_range, dtype):
    # Value-range sweep over the full float family: construction copies the
    # stored values verbatim, so every range (including the dtype-extreme
    # [0, max] / [min, 0] ranges) must round-trip exactly.
    variant, size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype, value_range)

    ref_out = _call_reference(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )
    res_out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    utils.gems_assert_close(res_out, ref_out, dtype)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_VALUE_CASES)
@pytest.mark.parametrize("value_range", _INT_VALUE_RANGES)
@pytest.mark.parametrize("dtype", _EXACT_COO_DTYPES)
def test_sparse_coo_tensor_int_value_ranges(case, value_range, dtype):
    # int/bool abs-exact path: the extreme [min, 0] / [0, max] ranges hit the
    # full integer span (int16 min through int64 max) with no wrap-around.
    variant, size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype, value_range)

    ref_out = _call_reference(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )
    res_out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(
        torch.ops.aten._values(res_out), torch.ops.aten._values(ref_out)
    )
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_nan_inf(dtype):
    # The factory copies the raw stored values and performs no arithmetic on
    # them, so inf/-inf/nan/-0.0 and the huge 1e30 magnitudes survive the
    # construction unchanged (1e30 covers the overflow-to-inf path in
    # fp16/bf16). equal_nan tolerates the nan outputs in every comparison.
    values = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = [[0, 1, 1, 0, 1, 0], [2, 0, 2, 1, 0, 2]]
    size = (2, 3)
    indices_t = _make_index_tensor(indices)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, len(indices[0]), dtype, 2, 0)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(
        torch.ops.aten._values(res_out),
        torch.ops.aten._values(ref_out),
        dtype,
        equal_nan=True,
    )
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_indices_size_zero_dim(dtype):
    # A logical size with a zero extent is valid: the result is an nnz == 0
    # coalesced tensor that still carries the full logical shape.
    size = (0, 4)
    indices_t = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, (), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, 2, 0)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_ndim():
    # indices must be 2-D (sparse_dim, nnz); a 1-D index tensor is rejected.
    indices_t = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(3, (), torch.float32)

    with pytest.raises(RuntimeError):
        _call_reference(indices_t, values, [3], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(indices_t, values, [3], torch.float32)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size():
    # A negative logical size is rejected (numel overflow); the candidate must
    # fail too rather than accept a nonsensical shape.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(2, (), torch.float32)

    with pytest.raises(RuntimeError):
        _call_reference(indices_t, values, [-2, 3], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(indices_t, values, [-2, 3], torch.float32)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_dtype():
    # The sparse COO layout requires int64 indices; an int32 index tensor is
    # rejected by the reference and must be by the candidate too.
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.int32, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    with pytest.raises(RuntimeError):
        _call_reference(indices_t, values, [2, 3], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(indices_t, values, [2, 3], torch.float32)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_nnz_mismatch():
    # indices and values must carry the same number of entries; a mismatch is
    # rejected by the reference and must be by the candidate too.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(3, (), torch.float32)

    with pytest.raises(RuntimeError):
        _call_reference(indices_t, values, [2, 3], torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(indices_t, values, [2, 3], torch.float32)


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_inferred_negative_index():
    # The size-inferred overload derives each sparse dim from max(index) + 1,
    # so a negative coordinate cannot be resolved and is rejected.
    indices_t = torch.tensor([[-1, 1]], dtype=torch.long, device=flag_gems.device)
    values = _make_values(2, (), torch.float32)

    with pytest.raises(RuntimeError):
        _call_reference(indices_t, values, None, torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(indices_t, values, None, torch.float32)
