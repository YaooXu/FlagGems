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
from . import test_utils as tu

# aten::detach_copy(Tensor self) -> Tensor returns a fresh, contiguous tensor
# holding the same values as ``self``. Unlike ``detach`` it never aliases the
# input (it is a real copy), and it is a pure memcpy, so every storage dtype
# (float / float8 / int / bool / complex) round-trips bit-exactly. detach_copy
# has NO autograd formula: calling torch.autograd.grad on its output raises
# RuntimeError, which test_detach_copy_no_backward pins as the reference
# contract.
#
# The .default overload is resolved through its public name "detach_copy" and
# the .out overload through "detach_copy.out" (KernelGen's
# override_gems_op("detach_copy", ...) / override_gems_op("detach_copy.out", ...)
# win over the direct callable, which is None today because flag_gems exposes no
# detach_copy kernel yet). Resolution happens inside each test, never at import
# time, so the process-local override installed by the harness is honored.
#
# Coverage follows the regular-operator spec (tests/test_utils.py):
#   * shape levels: tu.selected_shapes() (0~5 dims, quick/default via --quick);
#   * value ranges: tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over representative 0/1/3-dim shapes for every supported dtype;
#   * dtypes: the 9 spec dtypes the op accepts (int8, uint8, float8_e4m3fn,
#     float8_e5m2, float32, bfloat16, float16, int32, int64), all of which
#     tu.supported_dtypes reports for this device, plus float64, int16, bool and
#     complex64 which it also supports;
#   * edge cases: non-contiguous (transposed) inputs, empty tensors, nan/inf
#     and signed zero, and storage independence of the copy;
#   * backward: detach_copy has no autograd formula, so the negative contract
#     (autograd.grad raises) is pinned instead of a gradient comparison;
#   * negative: non-tensor input and a dtype-mismatched .out buffer must raise.
_DETACH_COPY_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_DETACH_COPY_INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
# The probe reports both fp8 formats as supported on this device; they need a
# dedicated bit-exact comparison because torch.testing.assert_close cannot
# compare float8 tensors on CPU.
_DETACH_COPY_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_DETACH_COPY_DTYPES = (
    _DETACH_COPY_FLOAT_DTYPES
    + _DETACH_COPY_INT_DTYPES
    + _DETACH_COPY_FP8_DTYPES
    + utils.BOOL_TYPES
    + [torch.complex64]
)

# Full shape × range grid for the selected level.
_DETACH_COPY_RANGE_SHAPES = tu.selected_shapes()
# Transposed views have non-unit strides; the kernel must honor them and emit a
# contiguous copy of the logical (non-contiguous) data.
_DETACH_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
# Degenerate/empty shapes: the copy must produce an empty contiguous tensor.
_DETACH_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
# Backward check shapes stay small (the reference autograd graph is tiny).
_DETACH_COPY_NO_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]
# Storage-independence shapes: overwriting the copy must not touch the input.
_DETACH_COPY_STORAGE_SHAPES = [(16, 32), (64, 128)]


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "detach_copy", getattr(flag_gems, "detach_copy", None)
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    # detach_copy returns a NEW contiguous tensor with the same logical values:
    # same shape/dtype/strides as the reference copy, never aliasing the input.
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    # The input must be untouched by the copy.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    # Clone so the post-call equality check can detect any mutation of the input
    # even when the reference runs on the same device.
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_value_ranges(shape, value_range, dtype):
    # Value-range coverage from the regular-operator spec: negative/positive
    # halves, dtype extremes and degenerate constant ranges. A pure copy must be
    # exact over all of them.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", tu.selected_cases(_DETACH_COPY_FLOAT_DTYPES))
def test_detach_copy_special_values(dtype):
    # nan/inf/-inf must survive a memcpy untouched, and -0.0 must keep its sign
    # bit. 1e30/-1e30 additionally overflow to +/-inf in fp16 on input
    # creation, which is fine: the copy still transfers the stored value.
    inp = torch.tensor(
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
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_equal(res_out, ref_out)
    # -0.0 must copy with its sign bit intact (equal_nan-tolerant compares treat
    # -0.0 == 0.0, so pin the sign explicitly).
    assert torch.equal(torch.signbit(res_out), torch.signbit(ref_out))


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp.clone())

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op()(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", _DETACH_COPY_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out_value_ranges(shape, value_range, dtype):
    # The .out path must reproduce the same values over every spec range while
    # writing into the caller's buffer (overwriting its previous value).
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op()(inp, out=res_out)

    assert ref_ret is ref_out
    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_non_contiguous(shape, dtype):
    # Transposed view input: the copy must materialize the logical values into a
    # fresh contiguous tensor honoring the non-unit strides. Slice the base
    # tensor symmetrically on both devices so the layouts match.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base.clone())
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_empty(shape, dtype):
    # Zero-element tensors must be handled without out-of-bounds accesses and
    # still yield an empty contiguous tensor of the right dtype.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_STORAGE_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    # Storage independence: mutating the copied output must leave the input
    # completely unaffected.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_equal(res_out, ref_out)
    res_out.fill_(3.25)
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NO_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_detach_copy_no_backward(shape, dtype):
    # detach_copy has no autograd formula: the reference output is a non-leaf
    # graph node but torch.autograd.grad raises RuntimeError ("derivative ...
    # not implemented"). The candidate must reproduce the reference output.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = _resolve_gems_op()(inp)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
def test_detach_copy_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float fails schema matching); the
    # candidate must fail too rather than silently accept scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.detach_copy_out
def test_detach_copy_out_rejects_wrong_dtype():
    # The .out overload validates the caller's buffer dtype and must raise for a
    # mismatched buffer instead of silently casting.
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp.clone())
    ref_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)
    res_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy.out(ref_inp, out=ref_out_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, out=res_out_bad)


@pytest.mark.detach_copy
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DETACH_COPY_DTYPES))
)
def test_detach_copy_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    candidate = flag_gems.testing.resolve_gems_op(
        "detach_copy", getattr(flag_gems, "detach_copy", None)
    )
    expected = torch.ops.aten.detach_copy(reference)
    actual = candidate(inp)
    tu.assert_result_equal(actual, expected)
