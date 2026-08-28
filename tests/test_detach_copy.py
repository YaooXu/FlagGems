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

# aten::detach_copy(Tensor self) -> Tensor returns a NEW tensor with the same
# shape, dtype, values and layout as self but detached from the autograd graph.
# Unlike aten::detach it is a *copy* (view_copy tag), not an aliasing view, so
# the result never shares storage with the input and is always contiguous even
# for non-contiguous inputs. The op is elementwise and dtype-agnostic; dtype
# coverage below spans float, int and bool inputs.
DETACH_COPY_DTYPES = utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. detach_copy is not yet
    # registered as a flag_gems attribute, so a missing attribute must not raise
    # before resolve_gems_op can consult the override registry.
    return flag_gems.testing.resolve_gems_op(
        "detach_copy", getattr(flag_gems, "detach_copy", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_copy_semantics(res_out, ref_out, inp):
    # detach_copy is a copy, not a view: shape/dtype match the reference copy,
    # the result is contiguous, and it must not alias the input storage.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous()
    assert res_out.data_ptr() != inp.data_ptr()
    assert not res_out._is_view()


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DETACH_COPY_DTYPES)
def test_detach_copy(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_copy_semantics(res_out, ref_out, inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", [(32, 64, 128), (16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_detach_copy_non_contiguous(shape, dtype):
    # The input is a strided slice (non-contiguous); detach_copy materializes a
    # fresh contiguous tensor whose values match the sliced input. Slice on both
    # the test device and the reference device so the two inputs share the same
    # memory layout.
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[::2, :, ::3]
    ref_inp = ref_base[::2, :, ::3]

    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_copy_semantics(res_out, ref_out, inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", [(16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    # detach_copy must not alias the input: mutating the copied result after the
    # call must not change the input, and the copy must reflect the input's
    # values at the time of the call.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _gems_op()(inp)

    with torch.no_grad():
        res_out.add_(1.0)
        ref_out.add_(1.0)

    _assert_close(res_out, ref_out, dtype)
    _assert_close(inp, ref_inp, dtype)
