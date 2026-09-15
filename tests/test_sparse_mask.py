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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

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
# Broadcast does not apply: sparse_mask is a gather whose two operands must have
# identical shapes (see the negative tests), so the only broadcast-like case is
# the rejection of a shape mismatch. Backward applies to the dense float self
# path and is exercised separately.
#
# The op is resolved through flag_gems.testing.resolve_gems_op() inside each
# test (never at module import time) so that the process-local override
# installed by KernelGen for this run wins.

# ---------------------------------------------------------------------------
# Dtype coverage for a sparse COO mask.
# ---------------------------------------------------------------------------
# Required spec dtypes first (int8/uint8/fp8e4m3/fp8e5m2/fp32/bf16/fp16/int32/
# int64), followed by the shared float/int/bool families. fp8 self operands are
# rejected by the aten reference itself ("Promotion for Float8 Types is not
# supported, attempted to promote Float8_e4m3fn and Bool"). The successful
# cases therefore exclude FP8 and retain int8/uint8.
_SPARSE_MASK_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        tu.REQUIRED_DTYPES
        + utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + utils.BOOL_TYPES
    )
)


def _make_mask(shape, density=0.5):
    # Boolean mask with ~(1 - density) fraction of nonzero entries (density is
    # the keep-threshold: larger values keep fewer positions), converted to a
    # coalesced sparse COO tensor.
    return (torch.rand(shape, device=flag_gems.device) > density).to_sparse()


_SPARSE_MASK_DTYPES = [
    dtype
    for dtype in _SPARSE_MASK_DTYPE_CANDIDATES
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
]


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


def _value_range_cases():
    cases = []
    for dtype in _SPARSE_MASK_DTYPES:
        for value_range in tu.selected_ranges():
            for shape in tu.selected_shapes():
                cases.append((shape, value_range, dtype))
    return cases


# One (shape, value_range, dtype) combo per Workload. 5 ranges x 7 shapes per
# dtype already gives 35 cases, far above tu.MIN_CASES.
_SPARSE_MASK_VALUE_RANGE_CASES = _value_range_cases()


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "sparse_mask", getattr(flag_gems, "sparse_mask", None)
    )


def _assert_masked(res_out, ref_out):
    # The current cases gather from dense or coalesced sparse inputs, without
    # reduction: indices and stored values must match the reference exactly.
    assert res_out.layout == ref_out.layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    tu.assert_result_equal(res_out.indices(), ref_out.indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape,value_range,dtype", _SPARSE_MASK_VALUE_RANGE_CASES)
def test_sparse_mask_value_ranges(shape, value_range, dtype):
    # Dense self over the five spec value ranges via tu.make_input + sparse COO
    # mask (the common dense-to-sparse gather path). Values are gathered, never
    # combined, so the selected values reproduce the self's value range exactly.
    # Mask density drops for the largest shapes so nnz stays moderate.
    numel = math.prod(shape)
    keep_threshold = 0.5 if numel <= 4096 else 0.9
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    mask = _make_mask(shape, density=keep_threshold)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out)
    # The gather must not mutate either operand: the reference was computed on
    # a pristine clone, so any candidate mutation of ``inp``/``mask`` shows up
    # here.
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(mask, ref_mask)


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
    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("base_shape,shape", _SPARSE_MASK_NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_non_contiguous(base_shape, shape, dtype):
    # A non-contiguous dense self must be gathered by logical (strided) index
    # positions, not physical memory offsets. Slice on both the test device and
    # the reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()
    mask = _make_mask(shape)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask_out
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_out(shape, dtype):
    # The .out variant must write into and return the provided out tensor. The
    # mask is bool-valued, so empty_like must be given the self dtype.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    mask = _make_mask(shape)
    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    out = torch.empty_like(mask, dtype=dtype)
    ref_out = torch.empty_like(ref_mask, dtype=dtype)

    torch.ops.aten.sparse_mask.out(ref_inp, ref_mask, out=ref_out)
    res_ret = _resolve_gems_op()(inp, mask, out=out)

    assert res_ret is out
    _assert_masked(res_ret, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_NANINF_SHAPES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_SPARSE_MASK_DTYPES))
)
def test_sparse_mask_nan_inf(shape, dtype, scenario):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    specials = tu.make_special_input(dtype, scenario)
    n = min(inp.numel(), specials.numel())
    inp.flatten()[:n] = specials[:n]

    # Mask every position holding a special value (plus random extras) so the
    # nan/inf entries are guaranteed to flow into the result.
    mask_dense = torch.rand(shape, device=flag_gems.device) > 0.7
    mask_dense.flatten()[:n] = True
    mask = mask_dense.to_sparse()

    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = _resolve_gems_op()(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_sparse_mask_backward(shape, dtype):
    # Compare candidate and reference gradients at the same mask positions.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    mask = _make_mask(shape)
    dense_grad = tu.make_input(dtype, shape, ["-1", "1"])
    # grad_output for the sparse result: sparse COO with the mask's indices and
    # values zero outside the mask.
    grad_out = (dense_grad * mask.to_dense()).to_sparse()

    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)
    ref_grad_out = tu.to_reference(grad_out)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)[0]

    res_out = _resolve_gems_op()(inp, mask)
    _assert_masked(res_out, ref_out)

    # The candidate must retain the reference's autograd behavior.
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)[0]
    tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_shape_mismatch():
    # There is no broadcast for sparse_mask: self and mask must have identical
    # shapes, otherwise both the aten reference and the candidate must raise.
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    mask = _make_mask((4, 6))
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            tu.to_reference(self_t),
            tu.to_reference(mask),
        )
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
            tu.to_reference(self_t),
            tu.to_reference(dense_mask),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(self_t, dense_mask)
