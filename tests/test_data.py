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

# aten::data(Tensor self) -> Tensor is the deprecated ``Tensor.data`` accessor:
# it returns a new tensor that shares the input's storage (same data_ptr, shape,
# stride and storage_offset) and is detached from autograd (requires_grad=False,
# is_leaf=True, grad_fn=None), i.e. it behaves like detach() + view. No
# arithmetic happens at the call, so every storage dtype is supported and the
# observed values round-trip bit-for-bit.
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by TEST_LEVEL);
#   * value ranges: tu.selected_ranges() over representative ranks so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the storage-sharing alias round-trips all of them
#     exactly);
#   * edge cases: non-contiguous (strided) inputs, writing through the returned
#     alias, and nan/inf/±0.0 special values;
#   * autograd: the result is always detached — even a requires_grad input
#     yields a leaf that shares storage (broadcast/backward do not apply to a
#     unary detach-and-alias op, so they are not covered);
#   * negative: a non-tensor input raises on both the aten reference and the
#     candidate.
_DATA_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_DATA_RANGE_SHAPES = [(), (256,), (7, 13, 29)]
_DATA_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_DATA_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_DATA_AUTOGRAD_SHAPES = [(16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.data callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("data", getattr(flag_gems, "data", None))


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_data_semantics(res_out, ref_out, inp, dtype):
    # The observable result must match aten exactly: same shape/dtype on the
    # same device as the input, aliasing the input storage, and detached from
    # autograd.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    assert res_out.data_ptr() == inp.data_ptr()
    assert not res_out.requires_grad
    assert res_out.is_leaf
    assert res_out.grad_fn is None
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_data_semantics(res_out, ref_out, inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", _DATA_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip exactly through the aliased shallow copy.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.data
@pytest.mark.parametrize("shape", _DATA_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_non_contiguous(shape, dtype):
    # The zero-copy alias must preserve the exact layout of a non-contiguous
    # input: shape, stride, storage offset and the shared data pointer. Slice
    # on both the test device and the reference device so the two inputs share
    # the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.shape == inp.shape
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()
    assert not res_out.is_contiguous()
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.data
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_special_values(dtype):
    # data is a pure alias: +inf/-inf/nan/±0.0 round-trip unchanged; the
    # equal_nan=True comparison in assert_result_close tolerates the nan output.
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(values)

    assert res_out.data_ptr() == values.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.data
@pytest.mark.parametrize("shape", _DATA_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_data_mutation(shape, dtype):
    # The result shares storage with the input: mutating through the result
    # must be visible in the original tensor. The reference runs on an
    # independent clone so the two aliases are validated separately.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    res_out = _resolve_gems_op()(inp)
    ref_out = torch.ops.aten.data(ref_inp)

    res_out.add_(1.0)
    ref_out.add_(1.0)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.data
@pytest.mark.parametrize("shape", _DATA_AUTOGRAD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_data_autograd_detach(shape, dtype):
    # aten::data detaches from autograd: even when the input requires grad, the
    # result is a leaf that requires no grad while still aliasing the input
    # storage.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp)
    if ref_inp is inp:
        ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_data_semantics(res_out, ref_out, inp, dtype)


@pytest.mark.data
def test_data_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.data(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
