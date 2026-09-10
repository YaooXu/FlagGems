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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from . import test_utils as tu

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
# verbatim as qparams. Only the three integer storage dtypes are accepted.
#
# Spec dimension applicability:
# - Value ranges: the data path is a pure bit copy, so the main grid runs
#   tu.selected_ranges() over the accepted storage dtypes (the five spec ranges
#   are snapped into each dtype's domain). The qparams are the second value
#   dimension: scale/zero_point are stored verbatim, including non-finite
#   scales, which is the nan/inf dimension of the spec (no floating tensor
#   payload can exist because floating inputs are rejected).
# - Shape levels: the main grids sweep tu.selected_shapes() (the seven required
#   0~5-dim shapes) plus the (0,) empty grid. The qparams / boundary /
#   nan-inf tests pin their value dimension and use a few small fixed shapes.
# - Broadcast: N/A -- unary op with a single tensor input.
# - Backward: N/A -- the input is an integer storage tensor, no autograd graph.
# - Negative cases: non-storage input dtypes, a non-quantized .out buffer, a
#   wrong-dtype quantized .out buffer and a shape-mismatched .out buffer must
#   raise on the aten reference and the candidate must reject them too.

# Storage dtypes probed with tu.supported_dtypes + a custom probe (the default
# probe calls ``packet.default(x)`` with a single argument, but this op also
# needs (scale, zero_point), so the probe mirrors a real call). Probing the 9
# required spec dtypes plus float64/int16/bool reports exactly these three:
# every other dtype hits "Creation of quantized tensor requires quantized dtype
# like torch.quint8".
_DTYPE_CANDIDATE_NAMES = (
    "int8",
    "uint8",
    "float8_e4m3fn",
    "float8_e5m2",
    "float32",
    "bfloat16",
    "float16",
    "int32",
    "int64",
    "int16",
    "float64",
    "bool",
)
_DTYPE_CANDIDATES = [
    d
    for d in (getattr(torch, name, None) for name in _DTYPE_CANDIDATE_NAMES)
    if isinstance(d, torch.dtype)
]


def _probe_supported_dtypes():
    def _probe(op_name, dtype):
        packet = getattr(torch.ops.aten, op_name, None)
        if packet is None:
            return False
        try:
            x = torch.tensor([1, 2, 3], dtype=dtype, device=flag_gems.device)
            packet.default(x, 0.1, 0)
        except Exception:
            return False
        return True

    return tu.supported_dtypes(
        "_make_per_tensor_quantized_tensor",
        candidates=_DTYPE_CANDIDATES,
        probe=_probe,
    )


# Fall back to the known-good set if the probe cannot run (e.g. an environment
# where the aten packet is unavailable at import time): the reference rejects
# every dtype outside {uint8, int8, int32}.
_MAKE_PERTENSOR_INPUT_DTYPES = _probe_supported_dtypes() or [
    torch.uint8,
    torch.int8,
    torch.int32,
]

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
# Shape grid: the spec's shape levels (7 shapes at full level, the smoke shape
# at quick level) plus the empty grid, which the shared set does not contain --
# a pure copy kernel must still handle zero elements.
_GRID_SHAPES = tu.selected_shapes() + [(0,)]
# Fixed small shapes for the value-dimension tests (qparams / boundary /
# nan-inf): these pin scale/zero_point/data values rather than shape, so they
# stay level-independent and keep the collected-case budget above tu.MIN_CASES
# in both the quick and the full level.
_SMALL_SHAPES = [(7,), (4, 8), (2, 3, 5)]
# Boundary patterns filled into the boundary-value test tensor.
_BOUNDARY_PATTERNS = ("min_max_0_1", "constant_min", "constant_max")


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


def _boundary_input(dtype, pattern):
    info = torch.iinfo(dtype)
    if pattern == "min_max_0_1":
        values = [info.min, info.max, 0, 1]
        if dtype != torch.uint8:
            values.append(-1)
        tensor = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    elif pattern == "constant_min":
        tensor = torch.full((8,), info.min, dtype=dtype, device=flag_gems.device)
    else:
        tensor = torch.full((8,), info.max, dtype=dtype, device=flag_gems.device)
    # All patterns are 1-D; widen to 2-D so the op sees a non-trivial shape.
    return tensor.repeat(4, 1)


def _quant_buffer(shape, quant_dtype, scale, zero_point, device):
    return torch.ops.aten._empty_affine_quantized(
        shape, dtype=quant_dtype, device=device, scale=scale, zero_point=zero_point
    )


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
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__make_per_tensor_quantized_tensor_value_ranges(shape, dtype, value_range):
    # Main value-range x shape x storage-dtype grid (one workload per combo).
    # The data path is a bit copy, so the expected result is derived from the
    # aten reference for the same input.
    inp = _make_value_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = _resolve_gems_op()(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    # The input is only read; it must be untouched.
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", _SMALL_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_qparams(shape, dtype, scale, zero_point):
    # scale / zero_point are the second value dimension: they are stored
    # verbatim as qparams and never touch the data path.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(
        ref_inp, scale, zero_point
    )
    res_out = _resolve_gems_op()(inp, scale, zero_point)

    assert res_out.q_scale() == scale
    assert res_out.q_zero_point() == zero_point
    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("pattern", _BOUNDARY_PATTERNS)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_boundary_values(dtype, pattern):
    # make_tensor draws values strictly below the dtype max, so pin the exact
    # dtype bounds explicitly: min/max/0/(±1) must round-trip bit-exactly.
    inp = _boundary_input(dtype, pattern)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = _resolve_gems_op()(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("scale", _NON_FINITE_SCALES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_non_finite_scale(dtype, scale):
    # nan/inf dimension: the reference stores a non-finite scale verbatim (it
    # performs no validation), so the candidate must too.
    inp = _make_input((4, 8), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, scale, 0)
    res_out = _resolve_gems_op()(inp, scale, 0)

    _assert_scale(res_out.q_scale(), scale)
    _assert_scale(ref_out.q_scale(), scale)
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


# aten::_make_per_tensor_quantized_tensor.out(Tensor self, float scale, int
# zero_point, *, Tensor(a!) out) -> Tensor(a!) resets the qparams of the
# provided out tensor (keeping its shape and dtype) and returns the same object
# (alias semantics).
@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__make_per_tensor_quantized_tensor_out_value_ranges(shape, dtype, value_range):
    inp = _make_value_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    # The out buffers start with different qparams so the overwrite performed by
    # the op is observable. The out dtype must already be the derived quantized
    # dtype (the out overload cannot change the out tensor's dtype).
    ref_out_buf = _quant_buffer(shape, _QUANT_DTYPE[dtype], 1.0, 0, _ref_device())
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, 0.5, -3, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = _quant_buffer(shape, _QUANT_DTYPE[dtype], 1.0, 0, flag_gems.device)
    res_out = _resolve_gems_op_out()(inp, 0.5, -3, out=act_out_buf)
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_out_qparams(dtype, scale, zero_point):
    # The .out overload must overwrite the buffer's stale qparams (allocated
    # here with scale=1.0 / zero_point=0) with the requested ones.
    inp = _make_input((4, 8), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out_buf = _quant_buffer((4, 8), _QUANT_DTYPE[dtype], 1.0, 0, _ref_device())
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, scale, zero_point, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = _quant_buffer((4, 8), _QUANT_DTYPE[dtype], 1.0, 0, flag_gems.device)
    res_out = _resolve_gems_op_out()(inp, scale, zero_point, out=act_out_buf)
    assert res_out is act_out_buf

    assert res_out.q_scale() == scale
    assert res_out.q_zero_point() == zero_point
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

    ref_buf = _quant_buffer((2, 3), _WRONG_QUANT_DTYPE[dtype], 1.0, 0, _ref_device())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(
            ref_inp, 0.1, 0, out=ref_buf
        )

    act_buf = _quant_buffer((2, 3), _WRONG_QUANT_DTYPE[dtype], 1.0, 0, flag_gems.device)
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
    buf = _quant_buffer((2, 2), torch.quint8, 1.0, 0, flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(inp, 0.5, 0, out=buf)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve_gems_op_out()(inp, 0.5, 0, out=buf)
