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

# aten::detach_copy(Tensor self) -> Tensor returns a fresh, contiguous tensor
# holding the same values as ``self``. It never aliases the input (unlike
# detach, it is a real copy) and it is a pure memcpy, so every storage dtype
# (float/int/bool/complex) is bit-exact and the comparisons below are exact on
# the int/bool path and well inside the default float tolerance
# (equal_nan=True covers the nan/inf cases). The .default overload is resolved
# through its public name "detach_copy" (KernelGen's
# override_gems_op("detach_copy", ...) wins over the direct callable); the .out
# overload through "detach_copy.out" (the harness also registers the alias
# "detach_copy_out" so a default-less resolve falls back to the override).
# detach_copy has NO autograd formula: torch.autograd.grad on the output of a
# requires-grad input raises RuntimeError, pinned by test_detach_copy_no_backward.

_DETACH_COPY_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_DETACH_COPY_INT_DTYPES = utils.ALL_INT_DTYPES
# complex64 exercises the complex storage path (complex32 is not uniformly
# supported by make_input / assert helpers, so complex64 is the representative).
_DETACH_COPY_DTYPES = (
    _DETACH_COPY_FLOAT_DTYPES
    + _DETACH_COPY_INT_DTYPES
    + utils.BOOL_TYPES
    + [torch.complex64]
)

# Degenerate/empty shapes: the copy must produce an empty contiguous tensor
# without touching any data.
_DETACH_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
# Transposed views have non-unit strides; the kernel must honor them and emit a
# contiguous copy of the logical (non-contiguous) data.
_DETACH_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
# Backward check shapes stay small (the reference autograd graph is tiny).
_DETACH_COPY_NO_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]
# Storage-independence shapes: overwriting the copy must not touch the input.
_DETACH_COPY_STORAGE_SHAPES = [(16, 32), (64, 128)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. flag_gems exposes no
    # direct detach_copy callable today, so the default is None and the override
    # registry (or a LookupError) decides.
    return flag_gems.testing.resolve_gems_op(
        "detach_copy", getattr(flag_gems, "detach_copy", None)
    )


def _resolve_gems_op_out():
    # The ".out" overload is overridden by KernelGen under both "detach_copy.out"
    # and its alias "detach_copy_out"; the canonical name is "detach_copy.out".
    return flag_gems.testing.resolve_gems_op(
        "detach_copy.out", getattr(flag_gems, "detach_copy_out", None)
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    # detach_copy returns a NEW contiguous tensor with the same logical values:
    # same shape/dtype/strides as the reference copy, never aliasing the input.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous()
    if inp.numel() > 0:
        # A true copy must not share storage with the input.
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)
    # The input must be untouched by the copy.
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_value_ranges(shape, value_range, dtype):
    # Value-range coverage from the regular-operator spec: negative/positive
    # halves, dtype extremes and degenerate constant ranges. A pure copy must be
    # exact over all of them.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", _DETACH_COPY_FLOAT_DTYPES)
def test_detach_copy_special_values(dtype):
    # nan/inf/-inf must survive a memcpy untouched, and -0.0 must keep its sign
    # bit. 1e30/-1e30 additionally overflow to +/-inf in fp16/bf16 on input
    # creation, which is fine: the copy still transfers the stored value.
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)
    # -0.0 must copy with its sign bit intact (equal_nan-tolerant compares treat
    # -0.0 == 0.0, so pin the sign explicitly).
    assert torch.equal(torch.signbit(res_out), torch.signbit(ref_out))


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)
    # The input must be untouched.
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=res_out)

    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_non_contiguous(shape, dtype):
    # Transposed view input: the copy must materialize the logical values into a
    # fresh contiguous tensor honoring the non-unit strides.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_STORAGE_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    # Storage independence: mutating the copied output must leave the input
    # completely unaffected.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)
    res_out.fill_(3.25)
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NO_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_detach_copy_no_backward(shape, dtype):
    # detach_copy has no autograd formula: the reference raises RuntimeError for
    # requires-grad inputs. The candidate must reproduce the reference output.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = _resolve_gems_op()(inp)
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.detach_copy_negative
def test_detach_copy_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.detach_copy_out_negative
def test_detach_copy_out_rejects_wrong_dtype():
    # The .out overload validates the caller's buffer dtype and must raise for a
    # mismatched buffer instead of silently casting.
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)
    res_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy.out(ref_inp, out=ref_out_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op_out()(inp, out=res_out_bad)
