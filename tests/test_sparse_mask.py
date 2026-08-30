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

import math
import os
import sys

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package (and, for the sibling benchmark file, ``benchmark``) resolve
# to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::sparse_mask(Tensor self, Tensor mask) -> Tensor returns a sparse tensor
# that has the same indices/layout as ``mask`` and values gathered from ``self``
# at those positions. ``mask`` must have exactly the same shape as ``self``
# (there is no broadcasting: a shape mismatch raises RuntimeError) and must be a
# sparse tensor itself (a dense mask raises NotImplementedError on every
# backend). Either operand may be dense or sparse COO; the result inherits the
# mask's indices/layout and the self's dtype. The .out variant
# (aten::sparse_mask.out) writes into and returns the provided ``out`` tensor.
#
# Coverage follows the sparse-operator adaptation of the regular-operator spec:
# the gather tests run over value ranges (tu.make_input + tu.selected_ranges)
# and shape levels (tu.selected_shapes()), backward covers the differentiable
# float path, negative cases pin the no-broadcast / sparse-mask-only contract,
# and nan/inf values (which are gathered, never combined) must propagate to the
# output. Masks are built with ``.to_sparse()`` (which produces coalesced COO
# tensors), so the result is coalesced and ``.indices()`` / ``.values()`` are
# directly accessible. Every (shape, value_range, dtype) parametrization combo
# is one distinct workload.
#
# The op is resolved through flag_gems.testing.resolve_gems_op() inside each
# test (never at module import time) so that the process-local override
# installed by KernelGen for this run wins.
_SPARSE_MASK_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Dedicated structural shapes: element counts stay small (<= 420) for the
# structure-heavy tests (sparse self, non-contiguous self, .out variant) since
# the op only gathers values at the mask's nonzero positions.
_SPARSE_MASK_STRUCT_SHAPES = [
    (16,),
    (2, 3),
    (8, 8),
    (16, 32),
    (4, 8, 16),
    (3, 7, 5, 4),
]

# Backward and nan/inf stay on small shapes (dense gradient comparison and
# special-value propagation are elementwise checks).
_SPARSE_MASK_BACKWARD_SHAPES = [(16, 32), (4, 8, 16), (3, 7, 5, 4)]
_SPARSE_MASK_NANINF_SHAPES = [(16,), (8, 8), (3, 7, 5, 4)]

# (base shape, sliced shape) pairs for the non-contiguous self case; slicing the
# last dimension by 2 produces a strided (non-contiguous) dense self.
_SPARSE_MASK_NON_CONTIGUOUS_CASES = [
    ((4, 8, 16), (4, 8, 8)),
    ((6, 10), (6, 5)),
]


def _make_mask(shape, density=0.5):
    # Boolean mask with ~(1 - density) fraction of nonzero entries (density is
    # the keep-threshold: larger values keep fewer positions), converted to a
    # coalesced sparse COO tensor.
    return (torch.rand(shape, device=flag_gems.device) > density).to_sparse()


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


def _assert_masked(res_out, ref_out, ref_mask, dtype, equal_nan=False):
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
    # Values are gathered from self: exact for int/bool, tolerance for float
    # (equal_nan covers the nan/inf entries in the special-value test).
    if dtype.is_floating_point:
        utils.gems_assert_close(
            res_out.values(), ref_out.values(), dtype, equal_nan=equal_nan
        )
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_value_ranges(shape, value_range, dtype):
    # Dense self over the value-range framework + sparse COO mask (the common
    # dense-to-sparse gather path). Values are gathered, never combined, so the
    # selected values reproduce the self's value range exactly. Mask density
    # drops for the largest shapes so nnz stays moderate.
    numel = math.prod(shape)
    keep_threshold = 0.5 if numel <= 4096 else 0.9
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())
    mask = _make_mask(shape, density=keep_threshold)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out, ref_mask, dtype)
    # The gather must not mutate either operand: the reference was computed on
    # a pristine clone, so any candidate mutation of ``inp``/``mask`` shows up
    # here.
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_sparse_self(shape, dtype):
    # Sparse COO self + sparse COO mask: the result keeps the mask's indices
    # and layout while taking values from the sparse self operand (values at the
    # mask's index positions, not the mask's own values).
    dense = tu.make_input(dtype, shape, ["-1", "1"])
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
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
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
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_out(shape, dtype):
    # The .out variant must write into and return the provided out tensor. The
    # mask is bool-valued, so empty_like must be given the self dtype.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
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


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_NANINF_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_mask_nan_inf(shape, dtype):
    # Special values (nan/inf/-inf, -0.0, and 1e30/-1e30 which overflow to inf
    # in fp16/bf16) at masked positions must propagate to the output untouched:
    # the gather never combines values. equal_nan=True tolerates the nan
    # entries.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    specials = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    n = min(inp.numel(), specials.numel())
    inp.flatten()[:n] = specials[:n]

    # Mask every position holding a special value (plus random extras) so the
    # nan/inf entries are guaranteed to flow into the result.
    mask_dense = torch.rand(shape, device=flag_gems.device) > 0.7
    mask_dense.flatten()[:n] = True
    mask = mask_dense.to_sparse()

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out, ref_mask, dtype, equal_nan=True)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_mask_backward(shape, dtype):
    # sparse_mask is differentiable w.r.t. self: the gradient is dense, nonzero
    # exactly at the mask's positions (the backward gathers the grad_output at
    # the mask indices). Validate the reference gradient analytically, then
    # compare the candidate's gradient when it advertises autograd support.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    mask = _make_mask(shape)
    dense_grad = tu.make_input(dtype, shape, ["-1", "1"])
    # grad_output for the sparse result: sparse COO with the mask's indices and
    # values zero outside the mask.
    grad_out = (dense_grad * mask.to_dense()).to_sparse()

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_grad_out = utils.to_reference(grad_out)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)[0]

    # d(out)/d(self) gathers grad_out at the mask positions; grad_out is zero
    # outside the mask by construction, so the dense gradient equals
    # grad_out.to_dense(). This validates the reference autograd path itself.
    expected_in_grad = ref_grad_out.to_dense()
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward must match the reference...
    res_out = _resolve_gems_op()(inp, mask)
    _assert_masked(res_out, ref_out, ref_mask, dtype)

    # ...and, if the candidate advertises autograd support (the current direct
    # kernel returns a leaf sparse tensor, so res_out.requires_grad is False and
    # this branch is skipped), its gradient must match the reference gradient.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)[0]
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_shape_mismatch():
    # There is no broadcast for sparse_mask: self and mask must have identical
    # shapes, otherwise both the aten reference and the candidate must raise.
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    mask = _make_mask((4, 6))
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(utils.to_reference(self_t), utils.to_reference(mask))
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(self_t, mask)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_rejects_dense_mask():
    # The mask operand must be a sparse tensor: a dense (strided) mask is not an
    # aten::sparse_mask argument on any backend.
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    dense_mask = torch.rand(4, 5, device=flag_gems.device) > 0.5
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            utils.to_reference(self_t), utils.to_reference(dense_mask)
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(self_t, dense_mask)
