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

# aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim)
# -> Tensor(a!) resizes a sparse COO tensor in place to ``size`` with
# ``sparse_dim`` sparse and ``dense_dim`` dense dimensions, returning ``self``.
#
# Reference semantics asserted by this file:
#   * ``sparse_dim + dense_dim`` must equal ``len(size)``.
#   * If ``size``, ``sparse_dim`` and ``dense_dim`` all already match ``self``
#     the call is a no-op and the stored indices/values are preserved verbatim.
#   * For a non-empty tensor (nnz > 0) the number of sparse and dense dims must
#     stay unchanged and no sparse/dense dimension may shrink; the supported
#     growth of sparse dimension sizes preserves the stored indices and values
#     verbatim. (Dense-dimension *size* growth would leave the values storage
#     in an implementation-defined state, so it is not exercised here.)
#   * For an empty tensor (nnz == 0) any resize is allowed and the tensor stays
#     empty with the requested shape / sparse / dense split.
#
# The op is a pure structural mutation (no arithmetic), so the regular-operator
# spec's value-range / shape-level / dtype / nan-inf dimensions assert that the
# stored payload is carried verbatim. Broadcast does not apply (single input,
# no elementwise pairing) and backward does not apply (the op is not
# differentiable / has no autograd formula), so both are intentionally absent.
#
# Coverage:
#   * layouts: each (src_shape, src_sparse_dim, src_nnz, size, new_sparse_dim,
#     new_dense_dim) case below is a distinct structure -- 2-D, 1-D, all-sparse
#     3-D, hybrid (dense trailing dims), the no-op path, sparse-dimension growth
#     on a non-empty tensor, and free reshapes of the empty tensor;
#   * dtypes: every sparse COO storage dtype the runtime supports, including the
#     spec-required int8 / uint8 / float8_e4m3fn / float8_e5m2 (probed on the
#     active device before parametrization) plus fp16/fp32/bf16/fp64,
#     int16/int32/int64 and bool;
#   * value ranges: the shared five per-dtype ranges, plus an explicit
#     nan / inf / -inf / -0.0 payload case;
#   * shape levels: tu.selected_shapes() with a legal sparse/dense split
#     (quick/all selected by the pytest --quick flag);
#   * negative: the invalid parameter triples the reference rejects, plus a
#     dense (non-sparse) input that cannot be routed to the sparse kernel.
_RESIZE_CASES = [
    ((4, 5), 2, 3, [4, 5], 2, 0),
    ((4, 5, 6), 2, 3, [4, 5, 6], 2, 1),
    ((5,), 1, 2, [8], 1, 0),
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5), 2, 3, [8, 8], 2, 0),
    ((3, 4, 5), 3, 7, [4, 4, 5], 3, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
    ((4, 5), 2, 0, [2, 4, 5], 2, 1),
    ((4, 5), 2, 0, [4, 5, 6], 3, 0),
    ((4, 5), 2, 0, [3, 5], 2, 0),
    ((5,), 1, 0, [2, 3], 2, 0),
]

# Value-range dimension: an all-sparse and a hybrid (dense trailing dims) grow,
# so every shared value range is exercised on both storage layouts.
_VALUE_RANGE_CASES = [
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
]

# Dtype coverage: the spec-required 9 dtypes where the operator supports them.
# int8 / uint8 / fp8 are hard requirements when the CUDA kernel supports them,
# so they are probed on the active device rather than assumed. float8 dtypes are
# looked up defensively (older torch builds have no float8 attribute).
_EXTRA_DTYPES = [torch.int8, torch.uint8]
for _fp8_name in ("float8_e4m3fn", "float8_e5m2"):
    _fp8_dtype = getattr(torch, _fp8_name, None)
    if _fp8_dtype is not None:
        _EXTRA_DTYPES.append(_fp8_dtype)

_CANDIDATE_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + _EXTRA_DTYPES + utils.BOOL_TYPES
)


def _sparse_dtype_probe(op_name, dtype):
    # tu.supported_dtypes' default probe calls the operator with a *dense*
    # tensor, which sparse_resize_ rejects for every dtype; use a tiny 1-D
    # sparse COO tensor instead and treat any exception as "unsupported".
    try:
        index = torch.tensor([0, 1], dtype=torch.long, device=flag_gems.device)
        values = torch.ones((2,), dtype=torch.float32, device=flag_gems.device).to(
            dtype
        )
        inp = torch.sparse_coo_tensor(
            index.unsqueeze(0), values, (4,), device=flag_gems.device
        )
        torch.ops.aten.sparse_resize_(inp, [6], 1, 0)
        return True
    except Exception:
        return False


# Only the dtypes this device's sparse COO storage actually accepts.
_RESIZE_DTYPES = tu.supported_dtypes(
    "sparse_resize_", _CANDIDATE_DTYPES, probe=_sparse_dtype_probe
)

# Shape-level dimension: the shared selected_shapes (quick/all levels via
# --quick) minus the 0-dim scalar, which is not representable as a sparse
# COO tensor (at least one sparse dimension is required).
_SPARSE_SHAPE_LEVELS = tuple(s for s in tu.selected_shapes() if len(s) > 0)

# The error paths of sparse_resize_ are structural and dtype-independent; a
# float and an int path suffice to pin the raise behavior.
_NEGATIVE_DTYPES = [torch.float32, torch.int32]

# Negative cases: (src_shape, src_sparse_dim, nnz, bad_size, bad_sparse_dim,
# bad_dense_dim) tuples the reference rejects before touching storage.
_INVALID_RESIZE_CASES = [
    pytest.param(
        ((4, 5), 2, 3, [4, 5, 6], 2, 0),
        id="invalid_split",
    ),
    pytest.param(
        ((4, 5), 2, 3, [6, 5], 1, 1),
        id="sparse_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [6, 5, 6], 1, 2),
        id="dense_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [3, 5], 2, 0),
        id="shrink_sparse_dim_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [4, 5, 4], 2, 1),
        id="shrink_dense_dim_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [4, -5], 2, 0),
        id="negative_size",
    ),
]


def _num_sparse_positions(shape, sparse_dim):
    num_sparse = 1
    for d in shape[:sparse_dim]:
        num_sparse *= d
    return num_sparse


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Deterministic CPU-side generation of a sparse COO tensor with unique,
    # lexicographically sorted indices so the verbatim storage preservation of
    # resize can be asserted. nnz must not exceed the number of sparse
    # positions; for sparse_dim == 0 only nnz == 0 is representable.
    # ``values``, when given, overrides the deterministic random payload (used
    # by the value-range / nan-inf tests); it must have shape
    # ``(nnz,) + shape[sparse_dim:]``. The payload is always moved to the test
    # device before the sparse tensor is constructed, so a dtype only needs to
    # be constructible on the active (sparse-capable) backend.
    gen = torch.Generator("cpu").manual_seed(seed)
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    num_sparse = _num_sparse_positions(shape, sparse_dim)
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        # Generate in a wide dtype then cast: randn/randint do not accept every
        # storage dtype (notably fp8), and random_ cannot sample unsigned
        # ranges that include negative values.
        if dtype == torch.bool:
            values = torch.randint(
                0, 2, values_shape, generator=gen, device="cpu"
            ).bool()
        elif dtype.is_floating_point:
            values = torch.randn(values_shape, generator=gen, device="cpu").to(dtype)
        else:
            # Keep the magnitude small so the values stay valid for every
            # integer storage dtype (int8/int16 included).
            values = torch.randint(-5, 6, values_shape, generator=gen, device="cpu").to(
                dtype
            )
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
        device=flag_gems.device,
    )


def _nan_inf_values(dtype, values_shape, device):
    # A deterministic nan/inf/-inf/0/-0/finite pattern covering the non-finite
    # payloads a resize must carry verbatim (no arithmetic is performed).
    pattern = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5]
    numel = 1
    for d in values_shape:
        numel *= d
    flat = (pattern * (numel // len(pattern) + 1))[:numel]
    return torch.tensor(flat, dtype=dtype, device=device).reshape(values_shape)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_resize_ is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_resize_", getattr(flag_gems, "sparse_resize_", None)
    )


def _range_for_dtype(value_range, dtype):
    # For unsigned integer storage the shared [-1, 0] range collapses to the
    # single value 0, which torch.testing.make_tensor cannot sample
    # (from == to); use the equivalent constant-zero range instead.
    if (
        dtype != torch.bool
        and not dtype.is_floating_point
        and tu.dtype_bounds(dtype)[0] >= 0
        and value_range == ["-1", "0"]
    ):
        return ["0", "0"]
    return value_range


def _assert_sparse_structure(t, ref, size, nnz, dtype, sparse_dim, dense_dim):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # sparse/dense split, the nnz count and the indices/values storage shapes.
    assert t.layout == torch.sparse_coo
    assert ref.layout == torch.sparse_coo
    assert tuple(t.shape) == tuple(size)
    assert tuple(ref.shape) == tuple(size)
    assert t.dtype == dtype
    assert t.sparse_dim() == sparse_dim
    assert t.dense_dim() == dense_dim
    assert ref.sparse_dim() == sparse_dim
    assert ref.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(t) == nnz
    assert torch.ops.aten._nnz(ref) == nnz
    assert tuple(torch.ops.aten._indices(t).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._indices(ref).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._values(t).shape) == (nnz,) + tuple(size[sparse_dim:])
    assert tuple(torch.ops.aten._values(ref).shape) == (nnz,) + tuple(size[sparse_dim:])
    assert t.is_coalesced() == ref.is_coalesced()


def _assert_values_equal(t, ref, dtype):
    # resize preserves the stored entries verbatim (or leaves both tensors
    # empty). fp32/fp16/bf16/fp64 use the dtype-aware close helper; every other
    # storage dtype (ints, bool, fp8 -- which assert_close cannot compare) must
    # match bit-exactly.
    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(t, ref, dtype)
    else:
        utils.gems_assert_equal(t, ref)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_(case, dtype):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = _resolve_gems_op()(inp, size, new_sparse_dim, new_dense_dim)

    # In-place semantics: the op returns self and mutates self in place.
    assert res_out is inp
    assert ref_out is ref_inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_sparse_structure(
        inp, ref_inp, size, nnz, dtype, new_sparse_dim, new_dense_dim
    )
    _assert_values_equal(inp, ref_inp, dtype)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_resize_value_ranges(case, dtype, value_range):
    # Value-range dimension: the stored entries are drawn from the shared
    # per-dtype ranges (sign coverage, [0,max], [min,0] and, at the all level,
    # the constant ranges). resize performs no arithmetic, so the payload must
    # survive verbatim regardless of its magnitude or sign.
    value_range = _range_for_dtype(value_range, dtype)
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    dense_shape = tuple(src_shape[sparse_dim:])
    values = tu.make_input(dtype, (nnz,) + dense_shape, value_range)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = _resolve_gems_op()(inp, size, new_sparse_dim, new_dense_dim)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_sparse_structure(
        inp, ref_inp, size, nnz, dtype, new_sparse_dim, new_dense_dim
    )
    _assert_values_equal(inp, ref_inp, dtype)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("shape", _SPARSE_SHAPE_LEVELS)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_shape_levels(shape, dtype):
    # Shape-level dimension: a no-op resize (identical size and sparse/dense
    # split) on every shared shape level preserves the stored entries verbatim.
    sparse_dim = 1
    dense_dim = len(shape) - 1
    nnz = min(2, shape[0])
    inp = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, list(shape), sparse_dim, dense_dim)
    res_out = _resolve_gems_op()(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_sparse_structure(inp, ref_inp, shape, nnz, dtype, sparse_dim, dense_dim)
    _assert_values_equal(inp, ref_inp, dtype)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_resize_nan_inf(dtype):
    # nan/inf/-inf/-0.0 are ordinary payloads for this structural op: growing
    # the sparse dims must move them verbatim (no arithmetic is performed).
    src_shape, sparse_dim, nnz = (4, 5, 6), 2, 4
    values = _nan_inf_values(dtype, (nnz, 6), flag_gems.device)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5, 6], 2, 1)
    res_out = _resolve_gems_op()(inp, [6, 5, 6], 2, 1)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_sparse_structure(inp, ref_inp, (6, 5, 6), nnz, dtype, 2, 1)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _INVALID_RESIZE_CASES)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_invalid_raises(case, dtype):
    # Negative cases: invalid sparse/dense split, split changes on a non-empty
    # tensor, shrinking sparse/dense dims and a negative size are all rejected
    # by the reference before storage is touched; a candidate must fail loudly
    # too instead of silently accepting them.
    src_shape, sparse_dim, nnz, size, bad_sparse_dim, bad_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(ref_inp, size, bad_sparse_dim, bad_dense_dim)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, size, bad_sparse_dim, bad_dense_dim)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_dense_input_rejected(dtype):
    # A dense (non-sparse) input cannot be routed to the sparse resize kernel;
    # the reference raises NotImplementedError (a RuntimeError subclass) and
    # the candidate must reject it too.
    if dtype.is_floating_point:
        inp = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 5, (4, 5), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(utils.to_reference(inp.clone()), [6, 5], 2, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, [6, 5], 2, 0)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_uncoalesced(dtype):
    # (0, 0) appears twice, so the input is uncoalesced; growing the sparse
    # dims must move the stored entries verbatim (never coalesce them). The
    # payload comes from the shared value-range framework.
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = tu.make_input(dtype, (4,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_sparse_structure(inp, ref_inp, (6, 5), 4, dtype, 2, 0)
    _assert_values_equal(inp, ref_inp, dtype)
