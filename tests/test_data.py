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

# aten::data(Tensor self) -> Tensor returns a storage-sharing shallow copy of
# the input: the result aliases the input's storage (same data_ptr, shape,
# stride and storage_offset) and is detached from autograd (requires_grad=False,
# is_leaf=True, grad_fn=None), i.e. it behaves like detach() + view. No
# arithmetic is performed, so the result must match bit-for-bit and every
# dtype the op supports is covered.
_DATA_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)
_NON_CONTIGUOUS_SHAPES = [
    (8, 16),
    (32, 64),
    (8, 16, 32),
]


def _make_input(shape, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolution order: (1) the process-local override injected by KernelGen,
    # (2) the direct flag_gems.data callable, (3) LookupError. Resolved inside
    # each test (never at import time) so the injected candidate wins.
    return flag_gems.testing.resolve_gems_op("data", getattr(flag_gems, "data", None))


def _assert_data_semantics(res_out, ref_out, inp, dtype):
    # The observable result must match aten exactly: same shape/dtype, on the
    # same device as the input, aliasing the input storage, and detached from
    # autograd.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    assert res_out.data_ptr() == inp.data_ptr()
    assert not res_out.requires_grad
    assert res_out.is_leaf
    assert res_out.grad_fn is None
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.data
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_data_semantics(res_out, ref_out, inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", _NON_CONTIGUOUS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_data_non_contiguous(shape, dtype):
    # The zero-copy alias must preserve the layout of a non-contiguous input:
    # shape, stride, storage offset and the shared data pointer.
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[::2, 1::2]
    ref_inp = ref_base[::2, 1::2]

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.shape == inp.shape
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()
    assert not res_out.is_contiguous()
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.data
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_data_mutation(dtype):
    # The result shares storage with the input: mutating through the result
    # must be visible in the original tensor.
    inp = _make_input((64, 64), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    ref_out.add_(1.0)
    res_out.add_(1.0)

    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
        utils.gems_assert_close(inp, ref_inp, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
        utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.data
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_data_autograd_detach(dtype):
    # aten::data detaches from autograd: even when the input requires grad, the
    # result is a leaf that requires no grad while still aliasing the input
    # storage.
    inp = _make_input((64, 64), dtype).requires_grad_()
    ref_inp = utils.to_reference(inp)
    if ref_inp is inp:
        ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_data_semantics(res_out, ref_out, inp, dtype)
