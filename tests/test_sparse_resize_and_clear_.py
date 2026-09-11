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

# aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim,
# int dense_dim) -> Tensor(a!) resizes a sparse COO tensor in place to ``size``
# with ``sparse_dim`` sparse and ``dense_dim`` dense dimensions and then clears
# all stored entries (nnz becomes 0), returning ``self``.
#
# Because the clear discards the stored indices/values, every redistribution is
# legal: the sparse/dense split may change freely, dimensions may both grow and
# shrink, and the source nnz does not constrain the target. The reference still
# validates the requested (size, sparse_dim, dense_dim) triple, so the negative
# dimension below pins those rejection paths.
#
# The overload is purely structural (no arithmetic), so
#   * every sparse COO storage dtype the runtime supports (the spec's required
#     int8/uint8/fp8/fp16/fp32/bf16/int32/int64 set plus int16, float64 and
#     bool) is exercised;
#   * the regular-operator spec's value-range dimension draws the stored values
#     from tu.selected_ranges() (they are discarded by the clear, so the result
#     must be identical for every range);
#   * the shape-level dimension resizes a fixed source to every
#     tu.selected_shapes() level (ranks 0-5) with a legal sparse/dense split;
#   * nan / inf / -inf payloads are covered explicitly.
#
# No broadcast dimension applies (single input tensor) and no backward
# dimension applies (the op has no autograd support).


def _unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


# Sparse COO storage dtypes. int8/uint8/fp8 are part of the spec's required
# set; all of them were probed to be representable by a sparse COO tensor on
# the active backend.
_RESIZE_DTYPES = _unique(
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

# Floating dtypes that can represent nan/inf/-inf (fp8_e4m3fn has no inf).
_NAN_INF_DTYPES = _unique(
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
)

# Each case is
# (src_shape, src_sparse_dim, src_dense_dim, dst_shape, dst_sparse_dim,
#  dst_dense_dim, src_nnz).
#
# The layouts covered are: identical metadata, grow/shrink all-sparse, hybrid
# (dense trailing dims) unchanged/grow, sparse<->dense redistribution, 1-D/4-D/
# 5-D, an all-dense-view target (sparse_dim == 0), and the empty (nnz == 0)
# source.
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

# Arbitrary reshapes of an empty (nnz == 0) source: every target is legal.
_EMPTY_SOURCE_TARGETS = [
    ((7,), 1, 0),
    ((2, 3), 2, 0),
    ((4, 5, 6), 2, 1),
    ((4, 5), 0, 2),
    ((3, 3, 3, 3), 3, 1),
]

# Shape levels and value ranges from the regular-operator spec (quick/all
# selected by the pytest --quick flag, read at import time).
_SELECTED_SHAPES = tu.selected_shapes()
_SELECTED_RANGES = tu.selected_ranges()

_NEGATIVE_DTYPES = [torch.float32, torch.int8]

# Candidate invalid (size, sparse_dim, dense_dim) triples: the sparse/dense
# split must sum to the requested number of dims and every entry must be
# non-negative. Some triples (a dense-view target, a zero size) are accepted by
# the reference, so the candidates are filtered below by an actual reference
# call rather than hardcoded.
_INVALID_CALL_CANDIDATES = [
    pytest.param([4, 5], 1, 0, id="split_too_small"),
    pytest.param([4, 5], 2, 1, id="split_too_large"),
    pytest.param([4, 5], 3, 0, id="split_too_large_sparse"),
    pytest.param([4, 5], 0, 2, id="dense_view_target"),
    pytest.param([4, 5], -1, 2, id="negative_sparse_dim"),
    pytest.param([4, 5], 2, -1, id="negative_dense_dim"),
    pytest.param([4, -5], 2, 0, id="negative_size"),
    pytest.param([4, 0], 2, 0, id="zero_size"),
]


def _num_sparse_positions(shape, sparse_dim):
    num_sparse = 1
    for d in shape[:sparse_dim]:
        num_sparse *= d
    return num_sparse


def _default_values(dtype, values_shape, gen):
    # Values are always generated on CPU (torch.randn is not implemented for
    # fp8 on CUDA) and moved to the test device by _make_sparse_input.
    if dtype.is_floating_point:
        base = torch.randn(values_shape, dtype=torch.float32, generator=gen)
        return base.to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    if dtype == torch.uint8:
        return torch.randint(0, 6, values_shape, dtype=dtype, generator=gen)
    return torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices). nnz must not exceed the number
    # of sparse positions; for sparse_dim == 0 only nnz == 0 is representable.
    # ``values``, when given, overrides the default payload (used by the
    # value-range / nan-inf tests) and is already on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    num_sparse = _num_sparse_positions(shape, sparse_dim)
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        values = _default_values(dtype, values_shape, gen)
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _call_is_rejected(size, sparse_dim, dense_dim):
    # A negative case is only kept if the reference really rejects it (a dense
    # or zero size target is legal for some backends).
    try:
        inp = _make_sparse_input((4, 5), 2, 1, torch.float32)
        torch.ops.aten.sparse_resize_and_clear_(
            utils.to_reference(inp), list(size), sparse_dim, dense_dim
        )
        return False
    except Exception:
        return True


_INVALID_CALLS = [
    p
    for p in _INVALID_CALL_CANDIDATES
    if _call_is_rejected(p.values[0], p.values[1], p.values[2])
]


def _nan_inf_values(dtype, values_shape, device):
    # A deterministic nan/inf/-inf/0/-0/finite pattern covering the non-finite
    # payloads a resize+clear must discard (no arithmetic is performed).
    pattern = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5]
    numel = 1
    for d in values_shape:
        numel *= d
    flat = (pattern * (numel // len(pattern) + 1))[:numel]
    return torch.tensor(flat, dtype=dtype, device=device).reshape(values_shape)


def _split_for_shape(shape):
    # A legal (sparse_dim, dense_dim) split for a target ``shape`` covering
    # all-sparse, hybrid and dense-heavy layouts across the shape levels.
    ndim = len(shape)
    if ndim == 0:
        return 0, 0
    sparse_dim = (ndim + 1) // 2
    return sparse_dim, ndim - sparse_dim


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


def _applicable_value_ranges():
    # The five shared ranges are snapped to the dtype's bounds by
    # tu.make_input; combinations that cannot be represented (e.g. the
    # negative-only range for uint8) are dropped rather than silently
    # mis-tested.
    pairs = []
    for dtype in _RESIZE_DTYPES:
        for value_range in _SELECTED_RANGES:
            try:
                tu.make_input(dtype, (1,), value_range)
            except Exception:
                continue
            pairs.append((dtype, value_range))
    return pairs


_VALUE_RANGE_PAIRS = _applicable_value_ranges()
_VALUE_RANGE_IDS = [
    f"{str(dtype).replace('torch.', '')}-{'_'.join(value_range)}"
    for dtype, value_range in _VALUE_RANGE_PAIRS
]


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
@pytest.mark.parametrize("dst_shape,dst_spd,dst_dnd", _EMPTY_SOURCE_TARGETS)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_empty_source(dst_shape, dst_spd, dst_dnd, dtype):
    # An empty (nnz == 0) source may be reshaped to any size / sparse / dense
    # split; the result stays empty with the requested metadata.
    inp = _make_sparse_input((4, 5), 2, 0, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = _resolve_gems_op()(inp, list(dst_shape), dst_spd, dst_dnd)

    assert res_out is inp
    assert ref_out is ref_inp
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
    values = tu.make_input(dtype, (4,), _SELECTED_RANGES[0]).to(flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
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
@pytest.mark.parametrize("dtype,value_range", _VALUE_RANGE_PAIRS, ids=_VALUE_RANGE_IDS)
def test_sparse_resize_and_clear_value_ranges(dtype, value_range):
    # Value-range dimension: the stored entries are drawn from the shared
    # per-dtype ranges (sign coverage, [0,max], [min,0], and constant ranges).
    # The clear discards every payload, so all ranges must produce the same
    # empty result as the reference.
    values = tu.make_input(dtype, (5,), value_range)
    inp = _make_sparse_input((4, 5), 2, 5, dtype, values=values)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_empty_resized(ref_inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_sparse_resize_and_clear_nan_inf(dtype):
    # nan / inf / -inf stored values are discarded by the clear; the resized
    # tensor is empty and exactly matches the reference.
    values = _nan_inf_values(dtype, (6,), flag_gems.device)
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
    assert ref_out is ref_inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_empty_resized(ref_inp, (6, 5), 2, 0, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("shape", _SELECTED_SHAPES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_shape_levels(shape, dtype):
    # Shape-level dimension: a fixed non-empty source resized to every shape
    # level (quick/all, ranks 0-5) with a legal sparse/dense split; the clear
    # keeps the result empty regardless of the target.
    sparse_dim, dense_dim = _split_for_shape(shape)
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(shape), sparse_dim, dense_dim
    )
    res_out = _resolve_gems_op()(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_empty_resized(inp, shape, sparse_dim, dense_dim, dtype)
    _assert_empty_resized(ref_inp, shape, sparse_dim, dense_dim, dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", _INVALID_CALLS)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_and_clear_invalid_params(size, sparse_dim, dense_dim, dtype):
    # Negative cases: the sparse/dense split must sum to len(size) and every
    # entry must be non-negative; the reference raises RuntimeError before
    # storage is touched and the candidate must fail loudly too.
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            utils.to_reference(inp.clone()), size, sparse_dim, dense_dim
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, size, sparse_dim, dense_dim)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_non_sparse_input():
    # A dense (non-sparse) input cannot be routed to the sparse resize kernel;
    # the reference raises NotImplementedError (a RuntimeError subclass) and
    # the candidate must reject it too.
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            utils.to_reference(inp.clone()), [4, 5], 2, 0
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, [4, 5], 2, 0)
