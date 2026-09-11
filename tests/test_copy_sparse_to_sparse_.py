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

# aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False)
# -> Tensor(a!)
#
# Copies the sparse structure and stored entries of ``src`` into ``self`` (both
# sparse COO tensors), resizing ``self`` to ``src``'s shape / sparse_dim / nnz as
# needed, and returns ``self``. The copy is a verbatim transfer of the stored
# indices and values -- it never coalesces, broadcasts, or performs arithmetic.
#
# Shape levels: this is a fixed-layout operator, so the spec's dense 0~5-D
# shapes are realized as sparse COO layouts: 1-D all-sparse, 2-D COO, hybrid
# COO (dense trailing dims), 3-D/4-D all-sparse, and the empty (nnz == 0)
# boundary. Broadcast does not apply (``self`` and ``src`` must have a
# compatible sparse structure). Sparse COO autograd has no derivative for the
# entry transfer, so backward is not tested (see
# test_copy_sparse_to_sparse_rejects_backward).

# ---------------------------------------------------------------------------
# Dtype probing
# ---------------------------------------------------------------------------
# The required dtype grid (int8 / uint8 / fp8 / fp32 / bf16 / fp16 / int32 /
# int64) plus the extra storage dtypes this op supports. The probe calls the
# *reference* op with a tiny sparse input per dtype, so the selected set does
# not depend on the candidate being installed and unsupported dtypes are never
# advertised.

_FP8_DTYPES = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if d is not None
)

_REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    *[d for d in _FP8_DTYPES],
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

_EXTRA_DTYPES = [torch.float64, torch.int16, torch.bool]

# ---------------------------------------------------------------------------
# Sparse layouts: (shape, sparse_dim, nnz)
# ---------------------------------------------------------------------------
# Each triple is a distinct sparse COO layout. ``sparse_dim`` splits the shape
# into sparse / dense dims, so the list walks 1-D all-sparse, 2-D COO, hybrid
# (sparse_dim=1 and 2), 3-D / 4-D all-sparse, and the nnz == 0 boundary.

_SPARSE_LAYOUTS = [
    ((6,), 1, 4),  # 1-D all-sparse
    ((4, 5), 2, 3),  # 2-D COO
    ((8, 8), 2, 16),  # 2-D COO, more stored entries
    ((16, 32), 2, 64),  # 2-D COO, many stored entries
    ((4, 4), 1, 3),  # hybrid, sparse_dim == 1
    ((2, 4, 5), 2, 6),  # 3-D hybrid (dense_dim == 1)
    ((3, 4, 5, 6), 2, 12),  # 4-D hybrid (dense_dim == 2)
    ((4, 5, 6), 3, 7),  # 3-D all-sparse
    ((2, 3, 4, 5), 4, 9),  # 4-D all-sparse
    ((4, 5), 2, 0),  # empty (nnz == 0) boundary
]

# Representative subset for the value-range grid (keeps the case count bounded
# while still covering all-sparse / hybrid / empty layouts).
_VALUE_RANGE_LAYOUTS = [
    ((6,), 1, 4),
    ((4, 5), 2, 3),
    ((2, 4, 5), 2, 6),
    ((2, 3, 4, 5), 4, 9),
    ((4, 5), 2, 0),
]

_NAN_INF_LAYOUTS = [
    ((4, 5), 2, 3),
    ((2, 4, 5), 2, 6),
    ((4, 5, 6), 3, 7),
    ((2, 3, 4, 5), 4, 9),
]


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _make_indices(shape, sparse_dim, nnz, seed):
    """Deterministic *coalesced* COO indices (unique, lexicographically sorted).

    Uncoalesced inputs are built explicitly where the storage order matters.
    """
    gen = torch.Generator("cpu").manual_seed(seed)
    num_sparse = 1
    for dim in shape[:sparse_dim]:
        num_sparse *= dim
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long)
    linear = torch.sort(torch.randperm(num_sparse, generator=gen)[:nnz]).values
    return torch.stack(torch.unravel_index(linear, shape[:sparse_dim]), dim=0)


def _make_values(values_shape, dtype, seed=0, value_range=None):
    """Stored entries for ``values_shape`` / ``dtype``.

    ``value_range`` selects the regular-operator value-range framework
    (``tu.make_input``). Without it the entries come from randn / small random
    integers, matching the existing accuracy-test style.
    """
    if value_range is None:
        gen = torch.Generator("cpu").manual_seed(seed)
        if dtype.is_floating_point:
            return torch.randn(values_shape, dtype=torch.float32, generator=gen).to(
                dtype
            )
        if dtype == torch.bool:
            return torch.randint(0, 2, values_shape, dtype=torch.bool, generator=gen)
        low = 0 if dtype == torch.uint8 else -5
        return torch.randint(low, 6, values_shape, dtype=dtype, generator=gen)

    # Value-range framework. Seed the global RNG so make_tensor is reproducible.
    torch.manual_seed(seed)
    try:
        return tu.make_input(dtype, values_shape, list(value_range))
    except RuntimeError:
        # Unsigned integer dtypes clamp the negative bound symbols to 0; a range
        # that collapses after clamping is realized as the constant bound.
        bound_low, bound_high = tu.dtype_bounds(dtype)
        low = int(max(tu.resolve_bound(value_range[0], dtype), bound_low))
        high = int(min(tu.resolve_bound(value_range[1], dtype), bound_high))
        if low >= high:
            return torch.full(values_shape, low, dtype=dtype, device=flag_gems.device)
        return torch.randint(
            low, high + 1, values_shape, dtype=dtype, device=flag_gems.device
        )


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, value_range=None):
    indices = _make_indices(shape, sparse_dim, nnz, seed)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    values = _make_values(values_shape, dtype, seed=seed, value_range=value_range)
    return torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        values.to(flag_gems.device),
        tuple(shape),
        device=flag_gems.device,
    )


def _make_nan_inf_values(values_shape, dtype):
    # Repeating nan / +inf / -inf / finite pattern. A verbatim copy keeps every
    # stored entry identical, so the CPU reference and the device candidate
    # agree exactly (nan handled by equal_nan=True).
    base = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.5,
            -0.5,
            2.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            1.5,
        ],
        dtype=dtype,
    )
    numel = 1
    for dim in values_shape:
        numel *= dim
    return base.repeat((numel + base.numel() - 1) // base.numel())[:numel].view(
        values_shape
    )


def _probe_supported_dtypes():
    supported = []
    for dtype in _REQUIRED_DTYPES + _EXTRA_DTYPES:
        try:
            src = _make_sparse_input((4, 5), 2, 3, dtype)
            dst = torch.zeros_like(src)
            torch.ops.aten.copy_sparse_to_sparse_(dst, src, False)
        except Exception:
            continue
        supported.append(dtype)
    # Fall back to float32 so the module still collects on a backend where the
    # reference sparse op is unavailable; the tests then fail on the real call.
    return tuple(supported) or (torch.float32,)


_SUPPORTED_DTYPES = _probe_supported_dtypes()
_FLOAT_DTYPES = tuple(d for d in _SUPPORTED_DTYPES if d.is_floating_point) or (
    torch.float32,
)


# ---------------------------------------------------------------------------
# Candidate resolution and comparison helpers
# ---------------------------------------------------------------------------


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. Order:
    # (1) override, (2) the direct flag_gems callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "copy_sparse_to_sparse_",
        getattr(flag_gems, "copy_sparse_to_sparse_", None),
    )


def _assert_same_sparse(res, ref):
    assert res.layout == torch.sparse_coo
    assert ref.layout == torch.sparse_coo
    assert tuple(res.shape) == tuple(ref.shape)
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res._nnz() == ref._nnz()


def _assert_sparse_values_close(res, ref):
    """Compare two sparse COO tensors entry-by-entry.

    torch.testing's sparse path routes through ``index_add``, which is not
    implemented for float8, so the stored entries are compared directly (as
    float32 for the fp8 dtypes).
    """
    _assert_same_sparse(res, ref)
    torch.testing.assert_close(
        res._indices().detach().to("cpu"),
        ref._indices().detach().to("cpu"),
        rtol=0,
        atol=0,
    )
    res_val = res._values().detach().to("cpu")
    ref_val = ref._values().detach().to("cpu")
    if res.dtype in _FP8_DTYPES:
        res_val = res_val.to(torch.float32)
        ref_val = ref_val.to(torch.float32)
    tu.assert_result_close(res_val, ref_val)


# ---------------------------------------------------------------------------
# Core equivalence: candidate vs torch.ops.aten, for every layout and dtype
# ---------------------------------------------------------------------------


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _SPARSE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_copy_sparse_to_sparse_(layout, dtype, non_blocking):
    shape, sparse_dim, nnz = layout
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, non_blocking)
    res_out = _resolve_gems_op()(dst, src, non_blocking)

    # In-place semantics: the op returns self and mutates dst in place.
    assert res_out is dst
    assert ref_out is ref_dst
    # The destination adopts the source's shape, layout, dtype, and nnz.
    assert dst.layout == torch.sparse_coo
    _assert_same_sparse(res_out, ref_out)
    _assert_same_sparse(dst, src)
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)
    _assert_sparse_values_close(ref_dst, ref_src)


# ---------------------------------------------------------------------------
# Value-range coverage (regular-operator spec)
# ---------------------------------------------------------------------------


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _VALUE_RANGE_LAYOUTS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_copy_sparse_to_sparse_value_ranges(layout, dtype, value_range):
    shape, sparse_dim, nnz = layout
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype, value_range=value_range)
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert dst._nnz() == src._nnz()
    # A verbatim copy transfers the entries exactly for every value range.
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("layout", _NAN_INF_LAYOUTS)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_copy_sparse_to_sparse_nan_inf(layout, dtype):
    shape, sparse_dim, nnz = layout
    base = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    src = torch.sparse_coo_tensor(
        base._indices().clone(),
        _make_nan_inf_values(values_shape, dtype),
        tuple(shape),
        device=flag_gems.device,
    )
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert dst._nnz() == src._nnz()
    # Verbatim copy keeps nan / +inf / -inf entries identical; the comparison
    # uses equal_nan=True.
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


# ---------------------------------------------------------------------------
# Resize / structural boundaries
# ---------------------------------------------------------------------------


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_resizes_self(dtype):
    # self is smaller than src ((4, 5) vs (6, 5)) with a different nnz; the copy
    # must resize self in place to src's shape and nnz.
    src = _make_sparse_input((6, 5), 2, 8, dtype)
    dst = _make_sparse_input((4, 5), 2, 5, dtype, seed=1)
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == 5
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert tuple(dst.shape) == tuple(src.shape) == (6, 5)
    assert dst._nnz() == src._nnz() == 8
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_resizes_nnz(dtype):
    # Same shape on both sides but self stores more entries than src (7 vs 3):
    # the copy resizes only self's nnz, shrinking its storage.
    src = _make_sparse_input((4, 5), 2, 3, dtype)
    dst = _make_sparse_input((4, 5), 2, 7, dtype, seed=1)
    assert dst._nnz() == 7
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == src._nnz() == 3
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_grows_dense_dims(dtype):
    # same sparse dims but self has fewer dense columns ((4, 5, 2) vs
    # (4, 5, 3)): the copy grows the dense dimensions of self.
    src = _make_sparse_input((4, 5, 3), 2, 3, dtype)
    dst = _make_sparse_input((4, 5, 2), 2, 3, dtype, seed=1)
    assert tuple(dst.shape) == (4, 5, 2)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert tuple(dst.shape) == tuple(src.shape) == (4, 5, 3)
    assert dst.dense_dim() == src.dense_dim() == 1
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_empty_dst_adopts_sparse_dim(dtype):
    # An empty self may be resized to any structure: self has sparse_dim 2 with
    # an empty trailing dense dim while src is all-sparse 3-D with the same
    # logical shape, so the copy must rebuild self's sparse dims as well as nnz.
    shape = (2, 4, 5)
    src = _make_sparse_input(shape, 3, 4, dtype)
    dense_shape = tuple(shape[2:])
    indices = torch.empty((2, 0), dtype=torch.long, device=flag_gems.device)
    values = torch.empty((0,) + dense_shape, dtype=dtype, device=flag_gems.device)
    dst = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert dst.sparse_dim() == 2
    assert dst._nnz() == 0
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert dst.sparse_dim() == src.sparse_dim() == 3
    assert dst.dense_dim() == src.dense_dim() == 0
    assert dst._nnz() == src._nnz() == 4
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_empty_src(dtype):
    # An empty src must clear self to nnz == 0 while keeping the shape.
    src = _make_sparse_input((4, 5), 2, 0, dtype)
    dst = _make_sparse_input((4, 5), 2, 3, dtype, seed=1)
    assert dst._nnz() == 3
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == src._nnz() == 0
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_copy_sparse_to_sparse_uncoalesced(dtype):
    # (0, 0) appears twice, so the source is uncoalesced; the copy transfers the
    # stored indices and entries verbatim (never coalesces them).
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_values((4,), dtype, seed=0).to(flag_gems.device)
    src = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not src.is_coalesced()
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert dst._nnz() == src._nnz() == 4
    # Entry order is preserved too, so the indices can be compared directly.
    _assert_sparse_values_close(res_out, ref_out)
    _assert_sparse_values_close(dst, src)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

_NEGATIVE_DTYPE = torch.float32


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_self():
    # A dense (strided) self is out of contract; the reference asserts on it.
    # NotImplementedError is a RuntimeError subclass, TypeError covers a
    # candidate that rejects the argument type outright.
    src = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE)
    self_dense = torch.zeros((4, 5), dtype=_NEGATIVE_DTYPE, device=flag_gems.device)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(self_dense, src, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(self_dense, src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_src():
    src_dense = torch.randn((4, 5), dtype=_NEGATIVE_DTYPE, device=flag_gems.device)
    ref_dst = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE)
    res_dst = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE, seed=1)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, src_dense, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(res_dst, src_dense, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_csr():
    # Sparse CSR is not COO: dispatch rejects both self and src.
    csr = torch.randn(
        (4, 5), dtype=_NEGATIVE_DTYPE, device=flag_gems.device
    ).to_sparse_csr()
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(csr.clone(), csr, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(csr.clone(), csr, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_sparse_dim_change():
    # Resizing a non-empty sparse tensor to a different number of sparse dims is
    # unsupported: self.sparse_dim() must equal src.sparse_dim().
    ref_src = _make_sparse_input((2, 4, 5), 3, 3, _NEGATIVE_DTYPE)
    ref_dst = _make_sparse_input((2, 4, 5), 2, 3, _NEGATIVE_DTYPE)
    res_src = _make_sparse_input((2, 4, 5), 3, 3, _NEGATIVE_DTYPE, seed=1)
    res_dst = _make_sparse_input((2, 4, 5), 2, 3, _NEGATIVE_DTYPE, seed=2)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_shrinking_sparse_dims():
    # The sparse sizes of a non-empty self may only grow during the resize:
    # shrinking them (here (6, 5) -> (4, 5)) is unsupported.
    ref_src = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE)
    ref_dst = _make_sparse_input((6, 5), 2, 3, _NEGATIVE_DTYPE)
    res_src = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE, seed=1)
    res_dst = _make_sparse_input((6, 5), 2, 3, _NEGATIVE_DTYPE, seed=2)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_shrinking_dense_dims():
    # Dense dimensions of a non-empty self may only grow as well: shrinking them
    # (here dense size 3 -> 2) is unsupported.
    ref_src = _make_sparse_input((4, 5, 2), 2, 3, _NEGATIVE_DTYPE)
    ref_dst = _make_sparse_input((4, 5, 3), 2, 3, _NEGATIVE_DTYPE)
    res_src = _make_sparse_input((4, 5, 2), 2, 3, _NEGATIVE_DTYPE, seed=1)
    res_dst = _make_sparse_input((4, 5, 3), 2, 3, _NEGATIVE_DTYPE, seed=2)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises((RuntimeError, TypeError)):
        op(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_backward():
    # Sparse COO autograd has no formula for the raw entry transfer; a
    # differentiable call must fail loudly rather than silently drop the grad.
    src = _make_sparse_input((4, 5), 2, 3, _NEGATIVE_DTYPE)
    src.requires_grad_(True)
    dst = torch.zeros_like(src)
    op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError)):
        out = torch.ops.aten.copy_sparse_to_sparse_(dst.clone(), src, False)
        torch.autograd.grad(out.to_dense().sum(), [src], allow_unused=True)
    with pytest.raises((RuntimeError, TypeError)):
        out = op(torch.zeros_like(src), src, False)
        torch.autograd.grad(out.to_dense().sum(), [src], allow_unused=True)
