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
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import conftest as cfg  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_make_per_tensor_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_tensor_quantized_tensor`` and ``-m
# _make_per_tensor_quantized_tensor`` both work.
for _name in (
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int
# zero_point) -> Tensor wraps an integer tensor (the quantized int
# representation) into a per-tensor affine quantized tensor. The output dtype is
# derived from the input dtype via toQIntType (uint8 -> quint8, int8 -> qint8,
# int32 -> qint32); no quantization arithmetic is applied -- the output's
# int_repr is an exact copy of the input values and scale/zero_point are stored
# verbatim as qparams. Inputs of any other dtype are rejected by the aten
# reference ("Creation of quantized tensor requires quantized dtype like
# torch.quint8"), so only the three integer dtypes are covered.
#
# Regular-operator spec dimensions:
# - Value ranges: the data path is a pure bit copy, so every value in the
#   supported integer dtype produces the same observable result; the main tests
#   run tu.selected_ranges() (per-dtype bounds, sign coverage, constants) and a
#   dedicated boundary case pins the exact dtype min/max. The qparams are the
#   second value dimension -- scale/zero_point are stored verbatim, including
#   non-finite scales (the nan/inf dimension of the spec).
# - Shape levels: tu.selected_shapes() (0-dim scalar through 8-dim) plus the
#   (0,) empty grid.
# - Broadcast: N/A -- a unary op with a single tensor input.
# - Backward: N/A -- integer storage tensor, no autograd support.
# - Negative cases: non-storage input dtypes, non-quantized .out buffers,
#   wrong-dtype quantized .out buffers and a shape-mismatched .out buffer must
#   raise on the aten reference and the candidate alike.

_MAKE_PERTENSOR_INPUT_DTYPES = [torch.uint8, torch.int8, torch.int32]
_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}
# A different quantized dtype for each storage dtype, used to check that the
# .out overload rejects a buffer whose dtype does not match the derived one.
_WRONG_QUANT_DTYPE = {
    torch.uint8: torch.qint8,
    torch.int8: torch.quint8,
    torch.int32: torch.quint8,
}
# scale and zero_point are opaque qparams (the reference accepts any float /
# int), so representative values exercise the metadata path; the data path is a
# pure copy.
_MAKE_PERTENSOR_SCALES = [0.01, 0.5, 1.0]
_MAKE_PERTENSOR_ZERO_POINTS = [-1, 2]
# Non-finite scales are stored verbatim (the reference performs no validation);
# this is the nan/inf dimension of the regular-operator spec.
_NON_FINITE_SCALES = [float("nan"), float("inf"), float("-inf")]
# Every storage dtype outside {uint8, int8, int32} is rejected by the aten
# reference.
_REJECTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int16,
    torch.int64,
    torch.bool,
]
# The shared shape set misses zero-element tensors; a pure copy kernel must
# handle the empty grid case.
_MAKE_PERTENSOR_SHAPES = tu.selected_shapes() + [(0,)]


def _make_input(shape, dtype, device=None):
    # Full-range random values including the dtype max (randint's high is
    # exclusive, so info.max + 1 is required).
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min,
        info.max + 1,
        shape,
        dtype=dtype,
        device=flag_gems.device if device is None else device,
    )


def _make_value_input(dtype, shape, value_range):
    # tu.make_input resolves the spec's ranges per-dtype and delegates to
    # torch.testing.make_tensor, which for uint8 clamps negative bounds to 0
    # and then raises on the resulting degenerate randint range (from=0 >=
    # to=0). Resolve the bounds ourselves and clamp to the unsigned domain so
    # every selected_ranges() entry stays usable for uint8 too.
    if dtype == torch.uint8:
        low = max(int(tu.resolve_bound(value_range[0], dtype)), 0)
        high = max(int(tu.resolve_bound(value_range[1], dtype)), 0)
        low, high = sorted((low, high))
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.randint(low, high + 1, shape, dtype=dtype, device=flag_gems.device)
    return tu.make_input(dtype, shape, value_range)


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override installed by KernelGen for this run wins. The
    # default stays None until flag_gems._make_per_tensor_quantized_tensor is
    # registered; resolution order is: (1) override, (2) the direct flag_gems
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_make_per_tensor_quantized_tensor",
        getattr(flag_gems, "_make_per_tensor_quantized_tensor", None),
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_make_per_tensor_quantized_tensor.out",
        getattr(flag_gems, "_make_per_tensor_quantized_tensor_out", None),
    )


def _assert_scale(res_scale, ref_scale):
    # q_scale() round-trips the stored double; nan must compare via isnan.
    if math.isnan(ref_scale):
        assert math.isnan(res_scale)
    else:
        assert res_scale == ref_scale


def _assert_quant_metadata(res_out, ref_out, ref_inp, dtype):
    # _make_per_tensor_quantized_tensor wraps integer data in a fresh quantized
    # tensor: the observable contract is the derived output dtype, the stored
    # qparams, the shape, and the int representation (an exact copy of the input
    # values). The input is never mutated and the output never aliases it.
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == _QUANT_DTYPE[dtype]
    assert res_out.shape == ref_out.shape
    _assert_scale(res_out.q_scale(), ref_out.q_scale())
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    utils.gems_assert_equal(res_out.int_repr(), ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", _MAKE_PERTENSOR_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor(
    shape, dtype, value_range, scale, zero_point
):
    inp = _make_value_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(
        ref_inp, scale, zero_point
    )
    res_out = _resolve_gems_op()(inp, scale, zero_point)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    # The input is only read; it must be untouched.
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_boundary_values(dtype):
    # make_tensor draws values strictly below the dtype max, so pin the exact
    # dtype bounds explicitly: min/max/0/(±1) must round-trip bit-exactly.
    info = torch.iinfo(dtype)
    values = [info.min, info.max, 0, 1]
    if dtype != torch.uint8:
        values.append(-1)
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device).repeat(4, 1)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = _resolve_gems_op()(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _NON_FINITE_SCALES)
def test__make_per_tensor_quantized_tensor_non_finite_scale(dtype, scale):
    inp = _make_input((4, 8), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, scale, 0)
    res_out = _resolve_gems_op()(inp, scale, 0)

    _assert_scale(res_out.q_scale(), scale)
    _assert_scale(ref_out.q_scale(), scale)
    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


# aten::_make_per_tensor_quantized_tensor.out(Tensor self, float scale, int
# zero_point, *, Tensor(a!) out) -> Tensor(a!) resets the qparams of the
# provided out tensor (keeping its shape and dtype) and returns the same object
# (alias semantics).
@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("shape", _MAKE_PERTENSOR_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_out(
    shape, dtype, value_range, scale, zero_point
):
    inp = _make_value_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    # The out buffers start with different qparams so the overwrite performed by
    # the op is observable. The out dtype must already be the derived quantized
    # dtype (the out overload cannot change the out tensor's dtype).
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=_QUANT_DTYPE[dtype], device=_ref_device(), scale=1.0, zero_point=0
    )
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, scale, zero_point, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    res_out = _resolve_gems_op_out()(inp, scale, zero_point, out=act_out_buf)
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_non_contiguous(dtype):
    # The copy must read through arbitrary input strides and still emit a
    # contiguous output. Slice on both the test device and the reference device
    # so the two inputs share the same memory layout.
    base = _make_input((16, 8), dtype)
    ref_base = utils.to_reference(base)
    inp = base[:, ::2]
    ref_inp = ref_base[:, ::2]

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = _resolve_gems_op()(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


# ---------------------------------------------------------------------------
# Negative cases: each invalid request must raise on the aten reference and the
# candidate must reject it too rather than silently succeeding.
# ---------------------------------------------------------------------------


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__make_per_tensor_quantized_tensor_rejects_non_storage_dtype(dtype):
    # Only uint8/int8/int32 storage tensors can be wrapped; the aten reference
    # raises "Creation of quantized tensor requires quantized dtype like
    # torch.quint8" for every other dtype.
    inp = torch.tensor([1, 2, 3], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.1, 0)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op()(inp, 0.1, 0)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_out_rejects_non_quantized_buffer(dtype):
    # The .out overload cannot change the out tensor's dtype, so a plain (non-
    # quantized) buffer is rejected by the reference and must be by the
    # candidate too.
    inp = _make_input((2, 3), dtype)
    ref_inp = utils.to_reference(inp)

    ref_buf = torch.empty((2, 3), dtype=torch.float32, device=_ref_device())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(
            ref_inp, 0.1, 0, out=ref_buf
        )

    act_buf = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op_out()(inp, 0.1, 0, out=act_buf)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_out_rejects_wrong_quantized_dtype(dtype):
    # A quantized buffer of any other dtype (e.g. qint8 for a quint8 output) is
    # rejected as well.
    inp = _make_input((2, 3), dtype)
    ref_inp = utils.to_reference(inp)

    ref_buf = torch.ops.aten._empty_affine_quantized(
        (2, 3),
        dtype=_WRONG_QUANT_DTYPE[dtype],
        device=_ref_device(),
        scale=1.0,
        zero_point=0,
    )
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(
            ref_inp, 0.1, 0, out=ref_buf
        )

    act_buf = torch.ops.aten._empty_affine_quantized(
        (2, 3),
        dtype=_WRONG_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op_out()(inp, 0.1, 0, out=act_buf)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="CPU reference resizes the out buffer; only the CUDA reference rejects "
    "a .out size that does not match the buffer (resize_ is unimplemented on "
    "QuantizedCUDA)",
)
def test__make_per_tensor_quantized_tensor_out_rejects_shape_mismatch():
    inp = torch.randint(0, 100, (4, 4), dtype=torch.uint8, device=flag_gems.device)
    buf = torch.ops.aten._empty_affine_quantized(
        (2, 2), dtype=torch.quint8, device=flag_gems.device, scale=1.0, zero_point=0
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(inp, 0.5, 0, out=buf)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op_out()(inp, 0.5, 0, out=buf)
