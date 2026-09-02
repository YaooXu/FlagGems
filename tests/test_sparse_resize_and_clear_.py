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

# aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim,
# int dense_dim) -> Tensor(a!) resizes a sparse COO tensor in place to ``size``
# with ``sparse_dim`` sparse and ``dense_dim`` dense dimensions and then clears
# all stored entries (nnz becomes 0), returning ``self``. Because the clear
# discards the stored indices/values, every redistribution is legal: the
# sparse/dense split may change freely, dimensions may both grow and shrink,
# and the source nnz does not constrain the target. The overload is purely
# structural (no arithmetic), so every sparse COO storage dtype the runtime
# supports (all floats, ints, and bool) is exercised.
#
# Coverage:
#   * layouts: each (src_shape, src_sparse_dim, src_dense_dim, dst_shape,
#     dst_sparse_dim, dst_dense_dim, src_nnz) case is a distinct structure:
#     identical metadata, grow/shrink all-sparse, hybrid (dense trailing dims)
#     unchanged/grow, sparse<->dense redistribution, 1-D/4-D/5-D, an
#     all-dense-view target (sparse_dim == 0) and the empty (nnz == 0) source;
#   * shape levels: tu.selected_shapes() resizes a fixed source to every
#     quick/all shape level with a legal sparse/dense split;
#   * value ranges: tu.selected_ranges() builds the source storage over the
#     spec's per-dtype numeric ranges, and nan/inf/-inf values are covered
#     separately (all discarded by the clear);
#   * negative: sparse_dim + dense_dim != len(size), negative sparse/dense dim
#     or size entries, and non-sparse (dense) inputs are rejected.
#
# No broadcast or backward dimensions apply: the operator has a single input
# tensor and performs no arithmetic (there is nothing to broadcast against or
# differentiate).
_RESIZE_CASES = [
    ((4, 5), 2, 0, (4, 5), 2, 0, 5),
    ((4, 5), 2, 0, (4, 5), 1, 1, 5),
    ((2, 3), 2, 0, (4, 5), 2, 0, 4),
    ((5, 5), 2, 0, (2, 3), 2, 0, 7),
    ((4, 5, 6), 2, 1, (4, 5, 6), 2, 1, 5),
    ((4, 5, 6), 1, 2, (3, 6, 7), 1, 2, 4),
    ((4, 5, 6), 2, 1, (4, 5, 6), 1, 2, 5),
    ((5,), 1, 0, (7,), 1, 0, 3),
    ((2, 3, 4, 5), 2, 2, (2, 3, 4, 5), 2, 2, 6),
    ((2, 2, 2, 2, 2), 3, 2, (3, 3, 3, 2, 2), 3, 2, 8),
    ((4, 5), 2, 0, (4, 5), 0, 2, 5),
    ((2, 3, 4), 3, 0, (2, 3, 4), 2, 1, 6),
    ((4, 5), 2, 0, (4, 5), 2, 0, 0),
]

# The op performs no arithmetic: it only resizes the metadata and clears the
# storage, so every sparse COO storage dtype the runtime supports (all floats,
# ints, and bool) is exercised.
_RESIZE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Shape levels and value ranges from the regular-operator spec (quick/all
# selected by the pytest --quick flag (read at import time)).
_SELECTED_SHAPES = tu.selected_shapes()
_SELECTED_RANGES = tu.selected_ranges()

# Invalid (size, sparse_dim, dense_dim) parameter triples: the sparse/dense
# split must sum to the requested number of dims and every entry must be
# non-negative.
_INVALID_CALLS = [
    ([4, 5], 1, 0),
    ([4, 5], 2, 1),
    ([4, 5], 3, 0),
    ([4, 5], -1, 2),
    ([4, 5], 2, -1),
    ([4, -5], 2, 0),
]


def _split_for_shape(shape):
    # A legal (sparse_dim, dense_dim) split for a target ``shape`` covering
    # all-sparse, hybrid and dense-heavy layouts across the shape levels.
    ndim = len(shape)
    if ndim == 0:
        return 0, 0
    sparse_dim = (ndim + 1) // 2
    return sparse_dim, ndim - sparse_dim


def _num_sparse_positions(shape, sparse_dim):
    num_sparse = 1
    for d in shape[:sparse_dim]:
        num_sparse *= d
    return num_sparse


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, value_range=None):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices). nnz must not exceed the number
    # of sparse positions; for sparse_dim == 0 only nnz == 0 is representable.
    # The result is moved to the test device. When ``value_range`` is given,
    # the stored values are drawn from the spec's value-range framework
    # (tu.make_input); otherwise per-dtype default values are used.
    gen = torch.Generator("cpu").manual_seed(seed)
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(
            _num_sparse_positions(shape, sparse_dim),
            generator=gen,
            device="cpu",
        )[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if value_range is not None:
        values = tu.make_input(dtype, values_shape, value_range)
    elif dtype.is_floating_point:
        values = torch.randn(values_shape, dtype=dtype, generator=gen, device="cpu")
    elif dtype == torch.bool:
        values = torch.randint(
            0, 2, values_shape, dtype=dtype, generator=gen, device="cpu"
        )
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype (int16 included).
        values = torch.randint(
            -5, 6, values_shape, dtype=dtype, generator=gen, device="cpu"
        )
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_resize_and_clear_ is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_resize_and_clear_",
        getattr(flag_gems, "sparse_resize_and_clear_", None),
    )


def _assert_empty_resized(t, shape, sparse_dim, dense_dim, dtype):
    # The resize+clear contract: the target shape / sparse / dense split is
    # applied and the storage is cleared, so nnz == 0 with indices (sparse_dim,
    # 0) and values (0,) + dense_shape. An nnz == 0 sparse tensor is coalesced
    # by definition.
    assert t.layout == torch.sparse_coo
    assert tuple(t.shape) == tuple(shape)
    assert t.dtype == dtype
    assert t.sparse_dim() == sparse_dim
    assert t.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(t) == 0
    assert tuple(torch.ops.aten._indices(t).shape) == (sparse_dim, 0)
    assert tuple(torch.ops.aten._values(t).shape) == (0,) + tuple(shape[sparse_dim:])
    assert t.is_coalesced()


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_(case, dtype):
    src_shape, src_spd, src_dnd, dst_shape, dst_spd, dst_dnd, src_nnz = case
    inp = _make_sparse_input(src_shape, src_spd, src_nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = _resolve_gems_op()(inp, list(dst_shape), dst_spd, dst_dnd)

    # In-place semantics: the op returns self and mutates the input in place.
    assert res_out is inp
    assert ref_out is ref_inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_empty_resized(inp, dst_shape, dst_spd, dst_dnd, dtype)
    _assert_empty_resized(ref_inp, dst_shape, dst_spd, dst_dnd, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_uncoalesced(dtype):
    # (0, 0) appears twice, so the input is uncoalesced; the clear must discard
    # the duplicated entries just like any other storage (never coalesce them
    # into a non-empty result).
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((4,), dtype=dtype, generator=gen, device="cpu")
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (4,), dtype=dtype, generator=gen, device="cpu")
    else:
        values = torch.randint(-5, 6, (4,), dtype=dtype, generator=gen, device="cpu")
    inp = torch.sparse_coo_tensor(
        indices, values.to(flag_gems.device), (4, 5), device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_empty_resized(ref_inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("value_range", _SELECTED_RANGES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_value_ranges(value_range, dtype):
    # The stored values are drawn from the spec's per-dtype value ranges
    # (negative, positive, extreme and degenerate). The clear discards them, so
    # every range must produce an identical empty result.
    inp = _make_sparse_input((4, 5), 2, 5, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_empty_resized(ref_inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_resize_and_clear_nan_inf(dtype):
    # nan / inf / -inf stored values are discarded by the clear; the resized
    # tensor is empty and exactly matches the reference (nothing left to
    # compare numerically).
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 4]],
        dtype=torch.long,
        device=flag_gems.device,
    )
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_empty_resized(ref_inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("shape", _SELECTED_SHAPES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_shape_levels(shape, dtype):
    # A fixed non-empty source resized to every shape level (quick/all,
    # ranks 0-8) with a legal sparse/dense split; the clear keeps the result
    # empty regardless of the target.
    sparse_dim, dense_dim = _split_for_shape(shape)
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(shape), sparse_dim, dense_dim
    )
    res_out = _resolve_gems_op()(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    _assert_empty_resized(inp, shape, sparse_dim, dense_dim, dtype)
    _assert_empty_resized(ref_inp, shape, sparse_dim, dense_dim, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", _INVALID_CALLS)
def test_sparse_resize_and_clear_invalid_params(size, sparse_dim, dense_dim):
    # The reference and the candidate must reject invalid parameter triples
    # with the same RuntimeError contract (sparse_dim + dense_dim must equal
    # len(size), all entries non-negative).
    inp = _make_sparse_input((4, 5), 2, 3, torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            utils.to_reference(inp.clone()), size, sparse_dim, dense_dim
        )
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, size, sparse_dim, dense_dim)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_non_sparse_input():
    # A dense (non-sparse) input cannot be routed to the sparse resize kernel;
    # the reference raises NotImplementedError (a RuntimeError subclass) and the
    # candidate must reject it too.
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            utils.to_reference(inp.clone()), [4, 5], 2, 0
        )
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, [4, 5], 2, 0)
