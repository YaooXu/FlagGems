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
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_make_dual_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._make_dual_copy`` and ``-m
# _make_dual_copy`` both work.
setattr(
    pytest.mark,
    "_make_dual_copy",
    MarkDecorator(Mark("_make_dual_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_make_dual_copy_out",
    MarkDecorator(Mark("_make_dual_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_make_dual_copy(Tensor(a) self, Tensor tangent, int level) -> Tensor is
# the view_copy materialization of _make_dual: instead of returning an aliasing
# view of ``self`` it materializes a fresh contiguous copy of the primal at the
# forward-mode AD level ``level``. Unlike _make_dual the copy is an ordinary
# tensor with no dual component, so unpacking the result with
# ``torch.autograd.forward_ad.unpack_dual`` must yield no tangent at all. The
# native implementation is guarded by an InferenceMode assert, so it is only
# callable on inference tensors; in that execution path aten ignores both
# ``tangent`` and ``level`` and simply returns the copy, which the candidate
# must reproduce. The op never mutates ``self`` and the result must not alias
# it.
#
# Adaptation notes for the regular-operator spec:
# - Broadcast: N/A -- the op takes a single ``self`` tensor (the tangent must
#   match self's size exactly; aten performs no broadcasting).
# - Backward: N/A -- the output is a fresh copy with no autograd support (the
#   op is tagged view_copy and builds no grad_fn), so there is no gradient to
#   compare.
# - Value ranges: covered below via tu.make_input + tu.selected_ranges(); the
#   copy must be bit-exact for every storage range and dtype family.
# - nan/inf: covered by the dedicated special-values test (a pure copy
#   round-trips nan/inf/signed-zero bit-for-bit).
# - Negative cases: missing/non-integer level, non-tensor input, and .out
#   dtype/shape validation are all asserted to fail on both sides.
_MAKE_DUAL_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES + [torch.complex64]
)
_MAKE_DUAL_COPY_LEVELS = [0, 1]


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_make_dual_copy" and "_make_dual_copy.out". flag_gems may not
    # register the underscore-prefixed attribute yet, so getattr supplies a
    # safe default and resolve_gems_op falls back to the package namespace.
    return flag_gems.testing.resolve_gems_op(
        "_make_dual_copy", getattr(flag_gems, "_make_dual_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_make_dual_copy.out", getattr(flag_gems, "_make_dual_copy_out", None)
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype):
    # _make_dual_copy returns a fresh contiguous copy: same shape/dtype, no
    # aliasing of the input, and the input is never mutated.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_dual_stripped(res_out, ref_out, dtype):
    # The view_copy materialization strips the forward-mode dual: unpacking the
    # result must yield no tangent on either side and plain primal values.
    res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
    ref_primal, ref_tangent = torch.autograd.forward_ad.unpack_dual(ref_out)
    assert res_tangent is None
    assert ref_tangent is None
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_primal, ref_primal, dtype)
    else:
        utils.gems_assert_equal(res_primal, ref_primal)


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
@pytest.mark.parametrize("level", _MAKE_DUAL_COPY_LEVELS)
def test__make_dual_copy_value_ranges(shape, value_range, dtype, level):
    # The result is the deterministic elementwise copy of the input no matter
    # what values the storage holds, so every range from the regular-operator
    # spec is exercised here (this is the value-range migration of the original
    # randn-based workload). tu.make_input produces an inference tensor, which
    # the aten reference requires.
    with torch.inference_mode():
        inp = tu.make_input(dtype, shape, value_range)
        tangent = tu.make_input(dtype, shape, value_range)
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)
        _assert_dual_stripped(res_out, ref_out, dtype)


@pytest.mark._make_dual_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
def test__make_dual_copy_out(shape, dtype):
    # The .out overload must write into and return the caller's buffer,
    # overwriting any previous contents.
    with torch.inference_mode():
        inp = tu.make_input(dtype, shape, ["-1", "1"])
        tangent = tu.make_input(dtype, shape, ["-1", "1"])
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())

        # Garbage-prefilled out buffers: the .out overload must overwrite them.
        out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)
        ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)

        ref_ret = torch.ops.aten._make_dual_copy.out(
            ref_inp, ref_tangent, 0, out=ref_out
        )
        res_ret = _resolve_gems_op_out()(inp, tangent, 0, out=out)

        # The .out variant writes into and returns the out tensor itself.
        assert res_ret is out
        assert ref_ret is ref_out
        _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
        utils.gems_assert_equal(out, ref_out)


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_copy_special_values(dtype):
    # A pure copy preserves every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    with torch.inference_mode():
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
            dtype=dtype,
            device=flag_gems.device,
        )
        ref_inp = utils.to_reference(values.clone())
        tangent = torch.ones_like(values)
        ref_tangent = utils.to_reference(tangent.clone())

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, 0)
        res_out = _resolve_gems_op()(values, tangent, 0)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._make_dual_copy
@pytest.mark.parametrize("dtype", _MAKE_DUAL_COPY_DTYPES)
def test__make_dual_copy_non_contiguous(dtype):
    # The copy must read through arbitrary input strides and still emit a
    # contiguous output. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout (and the reference
    # itself sees a non-contiguous inference tensor).
    with torch.inference_mode():
        base = tu.make_input(dtype, (8, 8, 8), ["-1", "1"])
        ref_base = utils.to_reference(base)
        inp = base[:, ::2, ::2]
        ref_inp = ref_base[:, ::2, ::2]
        assert not inp.is_contiguous()
        assert not ref_inp.is_contiguous()
        tangent = tu.make_input(dtype, inp.shape, ["-1", "1"])
        ref_tangent = utils.to_reference(tangent.clone())

        ref_out = torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, 0)
        res_out = _resolve_gems_op()(inp, tangent, 0)

        _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._make_dual_copy
def test__make_dual_copy_rejects_missing_level():
    # The schema requires the int ``level``; calling without it fails at
    # binding time.
    with torch.inference_mode():
        inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4,), ["-1", "1"])
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual_copy(ref_inp, ref_tangent)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, tangent)


@pytest.mark._make_dual_copy
def test__make_dual_copy_rejects_non_integer_level():
    # The schema requires an int ``level``; a float is rejected at binding time.
    with torch.inference_mode():
        inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4,), ["-1", "1"])
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual_copy(ref_inp, ref_tangent, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, tangent, 1.5)


@pytest.mark._make_dual_copy
def test__make_dual_copy_rejects_non_tensor_input():
    # The aten op requires Tensor arguments; Python floats hit a different
    # overload and raise. The candidate must fail too rather than silently
    # accept scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual_copy(3.14, 3.14, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14, 3.14, 0)


@pytest.mark._make_dual_copy_out
def test__make_dual_copy_out_rejects_wrong_dtype():
    # The out buffer must share the input's dtype; aten raises at binding
    # time and the candidate must fail too.
    with torch.inference_mode():
        inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4,), ["-1", "1"])
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())
        out = torch.empty(4, dtype=torch.int32, device=flag_gems.device)
        ref_out = torch.empty(4, dtype=torch.int32, device=ref_inp.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual_copy.out(ref_inp, ref_tangent, 0, out=ref_out)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op_out()(inp, tangent, 0, out=out)


@pytest.mark._make_dual_copy_out
def test__make_dual_copy_out_resizes_wrong_shape():
    # A mismatched out buffer is resized by the autogen out wrapper (with a
    # deprecation warning) and the same tensor object is returned; the
    # candidate must reproduce that mutation/alias behavior.
    with torch.inference_mode():
        inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
        ref_inp = utils.to_reference(inp.clone())
        ref_tangent = utils.to_reference(tangent.clone())
        out = torch.empty(2, 2, dtype=torch.float32, device=flag_gems.device)
        ref_out = torch.empty(2, 2, dtype=torch.float32, device=ref_inp.device)

        ref_ret = torch.ops.aten._make_dual_copy.out(
            ref_inp, ref_tangent, 0, out=ref_out
        )
        res_ret = _resolve_gems_op_out()(inp, tangent, 0, out=out)

    assert ref_ret is ref_out
    assert res_ret is out
    assert res_ret.shape == ref_ret.shape == (4, 8)
    tu.assert_result_close(res_ret, ref_ret)
