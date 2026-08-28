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

# aten::sparse_mask(Tensor self, Tensor mask) -> Tensor returns a sparse
# tensor that has the same indices/layout as ``mask`` and values gathered from
# ``self`` at those positions. ``mask`` must have exactly the same shape as
# ``self``; either operand may be dense or sparse COO, and the result inherits
# the mask's layout and the self's dtype. The .out variant
# (aten::sparse_mask.out) writes into and returns the provided ``out`` tensor.
#
# Masks are built with ``.to_sparse()`` (which produces coalesced COO tensors),
# so the result is coalesced and ``.indices()`` / ``.values()`` are directly
# accessible. Every (shape, dtype) parametrization combo below is one distinct
# workload; element counts stay small (<= 420) since the op only gathers
# values at the mask's nonzero positions.
_SPARSE_MASK_SHAPES = [
    (16,),
    (2, 3),
    (8, 8),
    (16, 32),
    (4, 8, 16),
    (3, 7, 5, 4),
]

# Values are gathered, not combined, so every storage dtype the op supports is
# covered: floating point (compared with the dtype-appropriate tolerance) and
# integer/bool (compared exactly).
_SPARSE_MASK_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# (base shape, sliced shape) pairs for the non-contiguous self case; slicing
# the last dimension by 2 produces a strided (non-contiguous) dense self.
_SPARSE_MASK_NON_CONTIGUOUS_CASES = [
    ((4, 8, 16), (4, 8, 8)),
    ((6, 10), (6, 5)),
]


def _make_dense(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-5, 6, shape, dtype=dtype, device=flag_gems.device)


def _make_mask(shape):
    # Boolean mask with ~50% nonzero entries, converted to a coalesced sparse
    # COO tensor.
    return (torch.rand(shape, device=flag_gems.device) > 0.5).to_sparse()


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so that the
    # process-local override installed by KernelGen for this run wins.
    return flag_gems.testing.resolve_gems_op(
        "sparse_mask", getattr(flag_gems, "sparse_mask", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "sparse_mask.out", getattr(flag_gems, "sparse_mask_out", None)
    )


def _assert_masked(res_out, ref_out, ref_mask, dtype):
    # Structure: the result is a sparse COO tensor with the mask's indices.
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == dtype
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    # Indices are int64 and must match both the reference and the mask exactly.
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    utils.gems_assert_equal(res_out.indices(), ref_mask.indices())
    # Values are gathered from self: exact for int/bool, tolerance for float.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask(shape, dtype):
    # Dense self + sparse COO mask (the common dense-to-sparse gather path).
    inp = _make_dense(shape, dtype)
    mask = _make_mask(shape)
    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out, ref_mask, dtype)
    # The gather must not mutate either operand.
    assert res_out is not inp
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_sparse_self(shape, dtype):
    # Sparse COO self + sparse COO mask: the result keeps the mask's indices
    # and layout while taking values from the sparse self operand.
    dense = _make_dense(shape, dtype)
    inp = dense.to_sparse()
    mask = _make_mask(shape)
    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out, ref_mask, dtype)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("base_shape, shape", _SPARSE_MASK_NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_non_contiguous(base_shape, shape, dtype):
    # A non-contiguous dense self must be gathered by logical (strided) index
    # positions, not physical memory offsets. Slice on both the test device and
    # the reference device so the two inputs share the same memory layout.
    base = _make_dense(base_shape, dtype)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()
    mask = _make_mask(shape)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out, ref_mask, dtype)


@pytest.mark.sparse_mask_out
@pytest.mark.parametrize("shape", _SPARSE_MASK_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_out(shape, dtype):
    # The .out variant must write into and return the provided out tensor. The
    # mask is bool-valued, so empty_like must be given the self dtype.
    inp = _make_dense(shape, dtype)
    mask = _make_mask(shape)
    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)

    out = torch.empty_like(mask, dtype=dtype)
    ref_out = torch.empty_like(ref_mask, dtype=dtype)

    ref_ret = torch.ops.aten.sparse_mask.out(ref_inp, ref_mask, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, mask, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_masked(res_ret, ref_ret, ref_mask, dtype)
