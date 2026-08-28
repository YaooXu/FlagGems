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
# Each (src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim) case
# below is a distinct layout: 2-D, 1-D, all-sparse 3-D, hybrid (dense trailing
# dims), the no-op path, sparse-dimension growth on a non-empty tensor, and
# free reshapes of the empty tensor.
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

# resize moves the stored entries verbatim (or clears them for empty-input
# reshapes), so every sparse COO storage dtype the runtime supports is
# exercised.
_RESIZE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0):
    # Deterministic CPU-side generation of a *coalesced* sparse COO tensor
    # (unique, lexicographically sorted indices) so the verbatim storage
    # preservation of resize can be asserted. The result is moved to the test
    # device.
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
    # default stays None until flag_gems.sparse_resize_ is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_resize_", getattr(flag_gems, "sparse_resize_", None)
    )


def _assert_sparse_structure(t, ref, size, nnz, dtype, sparse_dim, dense_dim):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # sparse/dense split, the nnz count, the indices/values storage shapes and
    # the coalesced flag.
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
    # empty), so float storages compare exactly here; the dtype-aware close
    # helper is still used to keep the usual float comparison policy.
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
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_uncoalesced(dtype):
    # (0, 0) appears twice, so the input is uncoalesced; growing the sparse
    # dims must move the stored entries verbatim (never coalesce them).
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

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5], 2, 0)
    res_out = _resolve_gems_op()(inp, [6, 5], 2, 0)

    assert res_out is inp
    assert ref_out is ref_inp
    _assert_sparse_structure(inp, ref_inp, (6, 5), 4, dtype, 2, 0)
    _assert_values_equal(inp, ref_inp, dtype)
