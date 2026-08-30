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

import sys as _sys
from pathlib import Path as _Path

# pytest --import-mode=importlib imports this module as <pkg>.test_copy_sparse_to_sparse_,
# where <pkg> is the "tests" or "benchmark" package of the checkout that actually
# holds this file (the KernelGen verification harness stages a temp copy of the
# FlagGems tree). When the driving process also has a same-named package on
# sys.path (e.g. the KernelGen repo's own tests/ directory), a bare relative
# import below would bind to that foreign package instead. Put the checkout root
# of *this* file first in sys.path so the relative imports resolve to the
# support files (accuracy_utils/test_utils) that ship next to it.
_CHECKOUT_ROOT = _Path(__file__).resolve().parent.parent
if str(_CHECKOUT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CHECKOUT_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False)
# -> Tensor(a!) copies the sparse structure and values of ``src`` into ``self``
# (both sparse COO tensors), resizing ``self`` to ``src``'s shape and nnz as
# needed, and returns ``self``. Each (shape, sparse_dim, nnz) triple below is a
# distinct layout: 2-D, batched hybrid (dense trailing dims), all-sparse 3-D,
# sparse_dim=1 hybrid, and the empty (nnz == 0) case.
#
# Shape levels: sparse COO layouts from 2-D through 4-D (hybrid and all-sparse),
# including the nnz == 0 boundary. Broadcast does not apply to this in-place
# sparse-to-sparse copy (self and src must have compatible sparse structure, the
# copy never broadcasts), and sparse COO autograd has no formula for the raw
# entry transfer, so there is no broadcast/backward test.
_SPARSE_CASES = [
    ((4, 5), 2, 3),
    ((8, 8), 2, 16),
    ((16, 32), 2, 64),
    ((2, 4, 5), 2, 6),
    ((3, 4, 5, 6), 2, 12),
    ((4, 5, 6), 3, 7),
    ((4, 4), 1, 3),
    ((4, 5), 2, 0),
]

# The copy moves raw stored entries, so every sparse COO storage dtype the
# runtime supports (all floats, ints, and bool) is exercised.
_SPARSE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage (regular-operator spec). The copy is a verbatim transfer
# of stored entries (no arithmetic), so every range from ``tu.selected_ranges()``
# -- including the full-width ``[0, max]`` and ``[min, 0]`` -- stays exact for
# every float storage dtype. A representative subset of layouts (2-D, batched
# hybrid, all-sparse 3-D, empty) keeps the test count bounded.
_VALUE_RANGE_CASES = [
    ((4, 5), 2, 3),
    ((8, 8), 2, 16),
    ((3, 4, 5, 6), 2, 12),
    ((4, 5, 6), 3, 7),
    ((4, 5), 2, 0),
]


def _make_sparse_input(shape, sparse_dim, nnz, dtype, low=None, high=None, seed=0):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices) so the exact structural copy
    # semantics can be asserted. The result is moved to the test device.
    # ``low``/``high`` are the resolved value-range bounds for float values
    # (value-range tests); by default the values come from randn.
    gen = torch.Generator("cpu").manual_seed(seed)
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    num_sparse = 1
    for d in shape[:sparse_dim]:
        num_sparse *= d
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if dtype.is_floating_point:
        if low is None:
            values = torch.randn(values_shape, dtype=dtype, generator=gen, device="cpu")
        else:
            # Compute in float64 so the full-width [min, 0] / [0, max] ranges
            # cannot overflow a float32 intermediate, then cast to ``dtype``.
            values = (
                torch.rand(
                    values_shape, dtype=torch.float64, generator=gen, device="cpu"
                )
                * (high - low)
                + low
            ).to(dtype)
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


def _make_nan_inf_values(values_shape, dtype):
    # Repeating pattern of nan / +inf / -inf / finite values. A verbatim copy
    # keeps every stored entry bit-identical, so the CPU reference and the
    # device candidate agree exactly (verified with equal_nan=True).
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
    for d in values_shape:
        numel *= d
    return base.repeat((numel + base.numel() - 1) // base.numel())[:numel].view(
        values_shape
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.copy_sparse_to_sparse_ is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "copy_sparse_to_sparse_", getattr(flag_gems, "copy_sparse_to_sparse_", None)
    )


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("case", _SPARSE_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_copy_sparse_to_sparse_(case, dtype, non_blocking):
    shape, sparse_dim, nnz = case
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, non_blocking)
    res_out = _resolve_gems_op()(dst, src, non_blocking)

    # In-place semantics: the op returns self and mutates dst in place.
    assert res_out is dst
    assert ref_out is ref_dst
    # The destination takes the source's shape, layout, dtype, and nnz.
    assert dst.layout == torch.sparse_coo
    assert tuple(dst.shape) == tuple(src.shape)
    assert tuple(ref_dst.shape) == tuple(ref_src.shape)
    assert dst.dtype == src.dtype
    assert dst._nnz() == src._nnz()
    # Coalesced sources are copied exactly: structure and values match.
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(dst, src)
    utils.gems_assert_equal(ref_dst, ref_src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_copy_sparse_to_sparse_resizes_self(dtype):
    # self has a smaller shape than src ((4, 5) vs (6, 5)) with a different
    # nnz; the copy must resize self in place to src's shape/nnz. (The sparse
    # dims only grow here: shrinking the sparse dims of a non-empty sparse
    # tensor is unsupported by the reference, so this stays on the supported
    # direction.)
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
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_copy_sparse_to_sparse_resizes_nnz(dtype):
    # Same shape on both sides but self stores more entries than src (7 vs 3):
    # the copy must resize only the nnz of self (shrinking its storage) to
    # match src.
    src = _make_sparse_input((4, 5), 2, 3, dtype)
    dst = _make_sparse_input((4, 5), 2, 7, dtype, seed=1)
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == 7
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert tuple(dst.shape) == (4, 5)
    assert dst._nnz() == src._nnz() == 3
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_copy_sparse_to_sparse_empty_dst_adopts_sparse_dim(dtype):
    # An empty self may be resized to any structure: self is empty with
    # sparse_dim 2 while src is all-sparse 3-D with the same logical shape, so
    # the copy must rebuild self's sparse dims as well as its nnz.
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
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_copy_sparse_to_sparse_uncoalesced(dtype):
    # (0, 0) appears twice, so the source is uncoalesced; the copy must still
    # transfer the stored entries verbatim into self (never coalesce them).
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
    src = torch.sparse_coo_tensor(
        indices, values.to(flag_gems.device), (4, 5), device=flag_gems.device
    )
    assert not src.is_coalesced()
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    # Structural comparison is avoided for uncoalesced storage (the dense form
    # accumulates the duplicate entries); the logical content must match.
    utils.gems_assert_equal(res_out.to_dense(), ref_out.to_dense())
    utils.gems_assert_equal(dst.to_dense(), src.to_dense())


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("range_symbols", tu.selected_ranges())
def test_copy_sparse_to_sparse_value_ranges(case, dtype, range_symbols):
    shape, sparse_dim, nnz = case
    low = tu.resolve_bound(range_symbols[0], dtype)
    high = tu.resolve_bound(range_symbols[1], dtype)
    src = _make_sparse_input(
        shape, sparse_dim, nnz, dtype, low=float(low), high=float(high)
    )
    dst = torch.zeros_like(src)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())

    ref_out = torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    res_out = _resolve_gems_op()(dst, src, False)

    assert res_out is dst
    assert ref_out is ref_dst
    assert dst._nnz() == src._nnz()
    # A verbatim copy transfers the values exactly; assert_result_close handles
    # float comparisons and per-dtype tolerance (exact here) uniformly.
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_copy_sparse_to_sparse_nan_inf(case, dtype):
    shape, sparse_dim, nnz = case
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    src = torch.sparse_coo_tensor(
        src._indices().clone(),
        _make_nan_inf_values(values_shape, dtype),
        shape,
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
    # Verbatim copy keeps nan / +inf / -inf entries identical; assert_result_close
    # compares floats with equal_nan=True.
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(dst, src)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_self():
    # The op is a sparse-COO-to-sparse-COO copy; a dense (strided) self is out
    # of contract and the reference asserts on it. NotImplementedError is a
    # RuntimeError subclass, so the candidate is held to the same contract on
    # any backend.
    src = _make_sparse_input((4, 5), 2, 3, torch.float32)
    self_dense = torch.zeros((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.copy_sparse_to_sparse_(self_dense, src, False)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(self_dense, src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_dense_src():
    src_dense = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    ref_dst = _make_sparse_input((4, 5), 2, 3, torch.float32)
    res_dst = _make_sparse_input((4, 5), 2, 3, torch.float32, seed=1)
    with pytest.raises(RuntimeError):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, src_dense, False)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(res_dst, src_dense, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_csr():
    # Sparse CSR layout is not COO: both self and src are rejected by dispatch.
    csr = torch.randn(
        (4, 5), dtype=torch.float32, device=flag_gems.device
    ).to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten.copy_sparse_to_sparse_(csr.clone(), csr, False)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(csr.clone(), csr, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_sparse_dim_change():
    # Resizing a non-empty sparse tensor to a different number of sparse dims
    # is unsupported: self.sparse_dim() must equal src.sparse_dim().
    ref_src = _make_sparse_input((2, 4, 5), 3, 3, torch.float32)
    ref_dst = _make_sparse_input((2, 4, 5), 2, 3, torch.float32)
    res_src = _make_sparse_input((2, 4, 5), 3, 3, torch.float32, seed=1)
    res_dst = _make_sparse_input((2, 4, 5), 2, 3, torch.float32, seed=2)
    with pytest.raises(RuntimeError):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(res_dst, res_src, False)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_rejects_shrinking_sparse_dims():
    # The sparse sizes of a non-empty self may only grow during the resize:
    # shrinking them (here (6, 5) -> (4, 5)) is unsupported.
    ref_src = _make_sparse_input((4, 5), 2, 3, torch.float32)
    ref_dst = _make_sparse_input((6, 5), 2, 3, torch.float32)
    res_src = _make_sparse_input((4, 5), 2, 3, torch.float32, seed=1)
    res_dst = _make_sparse_input((6, 5), 2, 3, torch.float32, seed=2)
    with pytest.raises(RuntimeError):
        torch.ops.aten.copy_sparse_to_sparse_(ref_dst, ref_src, False)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(res_dst, res_src, False)
