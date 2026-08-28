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

# aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False)
# -> Tensor(a!) copies the sparse structure and values of ``src`` into ``self``
# (both sparse COO tensors), resizing ``self`` to ``src``'s shape and nnz as
# needed, and returns ``self``. Each (shape, sparse_dim, nnz) triple below is a
# distinct layout: 2-D, batched hybrid (dense trailing dims), all-sparse 3-D,
# sparse_dim=1 hybrid, and the empty (nnz == 0) case.
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


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices) so the exact structural copy
    # semantics can be asserted. The result is moved to the test device.
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
    # tensor is unsupported by the CUDA reference, so this stays on the
    # supported direction.)
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
