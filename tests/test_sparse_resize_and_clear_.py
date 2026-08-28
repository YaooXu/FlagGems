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

# aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim,
# int dense_dim) -> Tensor(a!) resizes a sparse COO tensor in place to ``size``
# with ``sparse_dim`` sparse and ``dense_dim`` dense dimensions and then clears
# all stored entries (nnz becomes 0), returning ``self``. Unlike sparse_resize_,
# the clear makes every redistribution legal: the sparse/dense split may change
# freely and dimensions may both grow and shrink for any input, because the
# stored indices/values are discarded rather than preserved. The overload is
# purely structural (no arithmetic), so every sparse COO storage dtype the
# runtime supports (all floats, ints, and bool) is exercised.
#
# Each (src_shape, src_sparse_dim, src_dense_dim, dst_shape, dst_sparse_dim,
# dst_dense_dim, src_nnz) case below is a distinct layout: identical metadata,
# grow/shrink all-sparse, hybrid (dense trailing dims) unchanged/grow,
# sparse<->dense redistribution, 1-D/4-D/5-D, an all-dense-view target
# (sparse_dim == 0) and the empty (nnz == 0) source.
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


def _num_sparse_positions(shape, sparse_dim):
    num_sparse = 1
    for d in shape[:sparse_dim]:
        num_sparse *= d
    return num_sparse


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices). nnz must not exceed the number
    # of sparse positions; for sparse_dim == 0 only nnz == 0 is representable.
    # The result is moved to the test device.
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
