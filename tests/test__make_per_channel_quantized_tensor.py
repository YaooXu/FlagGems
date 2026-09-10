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

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from . import test_utils as tu

# ``_make_per_channel_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_channel_quantized_tensor`` and ``-m
# _make_per_channel_quantized_tensor`` both work.
for _name in (
    "_make_per_channel_quantized_tensor",
    "_make_per_channel_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor
# zero_point, int axis) -> Tensor reinterprets a *plain integer* tensor
# (uint8/int8/int32) as the storage of a torch.per_channel_affine quantized
# tensor carrying per-channel scale/zero_point metadata. The output quantized
# dtype is derived from the storage dtype: uint8 -> quint8, int8 -> qint8,
# int32 -> qint32; the integer values are copied to the output storage
# unchanged. The scale dtype selects the qscheme:
#   * integer zero_points -> torch.per_channel_affine (float64 scales and
#     int64 zero_points as reported by the q_per_channel_* getters);
#   * floating zero_points -> torch.per_channel_affine_float_qparams (float32
#     scales and zero_points).
# Both the data path (bit-exact copy) and the metadata (stored verbatim) are
# exact, so every assertion is an equality check (utils.gems_assert_equal).
#
# Regular-operator spec dimensions:
# - Value ranges: the storage tensor, the scales and the zero_points are the
#   only value-carrying inputs; all three are driven by tu.selected_ranges()
#   (per-dtype bounds, sign coverage, degenerate constants).
# - Shape levels: tu.selected_shapes() (0-dim scalar through 5-dim) with a
#   valid axis per rank, plus a dedicated axis-semantics grid covering every
#   positive/negative axis position. The op is not rank-fixed.
# - Broadcast: N/A -- the op takes a single storage tensor plus 1-D metadata.
# - Backward: N/A -- quantized tensors carry no autograd support.
# - Negative cases: non-storage input dtypes, non-float or non-1-D scales,
#   non-1-D zero_points and mismatched metadata lengths must raise on the aten
#   reference and on the candidate alike.

_STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]

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

# The scale tensor may be float32 or float64; the quantizer canonicalizes both
# to float64 (per_channel_affine) or float32 (float_qparams).
_SCALE_DTYPES = [torch.float32, torch.float64]

# Every non-storage dtype is rejected by the aten reference.
_REJECTED_INPUT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int16,
    torch.int64,
    torch.bool,
]

# Non-finite qparams are accepted and stored verbatim (the nan/inf dimension).
_NON_FINITE = [float("nan"), float("inf"), float("-inf")]

# One valid axis per shared shape level (0-dim through 5-dim). 0-dim / 1-dim
# only have axis 0; higher ranks pick an interior/negative axis.
_AXIS_BY_RANK = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3}

_SHAPE_AXIS = [
    (tuple(shape), _AXIS_BY_RANK[len(shape)]) for shape in tu.selected_shapes()
]
if tu.LEVEL == "quick":
    # The quick smoke level uses the single shared shape, which would leave the
    # collected case count below tu.MIN_CASES; widen the grid with two extra
    # ranks (the full level already uses the seven shared shape levels).
    _SHAPE_AXIS += [((2, 3, 4), 1), ((4, 5), 0)]

# Axis-semantics grid: every valid positive and negative axis encoding.
_AXIS_SHAPES = (
    [((2, 3, 4), 1), ((2, 3, 4), -2), ((2, 3, 4), 0), ((7,), -1)]
    if tu.LEVEL == "quick"
    else [
        ((7,), 0),
        ((7,), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((2, 3, 4), 0),
        ((2, 3, 4), 1),
        ((2, 3, 4), 2),
        ((2, 3, 4), -2),
        ((2, 3, 4), -3),
        ((7, 13, 29), 2),
        ((7, 13, 29), -1),
        ((2, 3, 4, 5), 2),
        ((2, 3, 4, 5), -1),
        ((2, 3, 4, 5, 6), 3),
        ((2, 3, 4, 5, 6), -1),
    ]
)


def _resolve(name):
    # Resolved inside each test (never at module import time) so that the
    # process-local override installed by KernelGen via ``override_gems_op`` for
    # this run wins. The .out overload is resolved through its aten schema name
    # ("_make_per_channel_quantized_tensor.out") so the override key matches the
    # schema; its direct fallback callable carries the flag_gems underscore
    # suffix ("_make_per_channel_quantized_tensor_out"). Resolution order:
    # (1) override, (2) the direct flag_gems callable, (3) LookupError.
    default = getattr(flag_gems, name.replace(".", "_"), None)
    return flag_gems.testing.resolve_gems_op(name, default)


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _num_channels(shape, axis):
    # A 0-dim storage tensor carries a single channel.
    return 1 if len(shape) == 0 else shape[axis]


def _make_range_tensor(dtype, shape, value_range):
    # tu.make_input delegates to torch.testing.make_tensor, which clamps
    # negative bounds to 0 for uint8 and then raises on the resulting degenerate
    # randint range (from=0 >= to=0). Resolve and clamp the bounds ourselves so
    # every selected_ranges() entry stays usable for uint8 too.
    if dtype == torch.uint8:
        low = max(int(tu.resolve_bound(value_range[0], dtype)), 0)
        high = max(int(tu.resolve_bound(value_range[1], dtype)), 0)
        low, high = sorted((low, high))
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.randint(low, high + 1, shape, dtype=dtype, device=flag_gems.device)
    return tu.make_input(dtype, shape, value_range)


def _zero_point_bounds(dtype):
    if dtype == torch.uint8:
        return 0, 256
    if dtype == torch.int8:
        return -128, 128
    info = torch.iinfo(torch.int32)
    return info.min, info.max


def _make_metadata(shape, axis, storage_dtype, scale_dtype):
    # Representative metadata: positive scales and zero_points inside the
    # storage dtype's range.
    num_channels = _num_channels(shape, axis)
    scales = torch.rand(num_channels, dtype=scale_dtype, device=flag_gems.device) + 0.1
    low, high = _zero_point_bounds(storage_dtype)
    zero_points = torch.randint(
        low, high, (num_channels,), dtype=storage_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _make_float_metadata(shape, axis, zero_point_dtype):
    # A floating-point zero_point tensor selects the
    # per_channel_affine_float_qparams qscheme.
    num_channels = _num_channels(shape, axis)
    scales = (
        torch.rand(num_channels, dtype=torch.float32, device=flag_gems.device) + 0.1
    )
    zero_points = torch.rand(
        num_channels, dtype=zero_point_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _make_out_buffer(shape, axis, storage_dtype, device, reference=False):
    # The out buffer must already carry the derived quantized dtype. Its initial
    # metadata is deliberately different so the overwrite performed by the op is
    # observable.
    num_channels = _num_channels(shape, axis)
    scales = torch.full((num_channels,), 9.0, dtype=torch.float64, device=device)
    zero_points = torch.full((num_channels,), 9, dtype=torch.int64, device=device)
    if reference:
        scales = utils.to_reference(scales)
        zero_points = utils.to_reference(zero_points)
    return torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=_QUANT_DTYPE[storage_dtype],
        device=device,
    )


def _assert_per_channel_affine(
    res_out, ref_out, inp, scales, zero_points, axis, equal_nan=False
):
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == _QUANT_DTYPE[inp.dtype]
    assert res_out.shape == ref_out.shape
    assert res_out.numel() == ref_out.numel()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.qscheme() == ref_out.qscheme() == torch.per_channel_affine
    assert res_out.q_per_channel_axis() == ref_out.q_per_channel_axis() == axis

    # The getters always report float64 scales and int64 zero_points.
    assert (
        res_out.q_per_channel_scales().dtype
        == ref_out.q_per_channel_scales().dtype
        == torch.float64
    )
    assert (
        res_out.q_per_channel_zero_points().dtype
        == ref_out.q_per_channel_zero_points().dtype
        == torch.int64
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        ref_out.q_per_channel_scales(),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        ref_out.q_per_channel_zero_points(),
        equal_nan=equal_nan,
    )

    # The stored metadata must reproduce the caller-supplied values exactly
    # (scales widened to float64, zero_points to int64).
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        utils.to_reference(scales).to(torch.float64),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        utils.to_reference(zero_points).to(torch.int64),
        equal_nan=equal_nan,
    )

    # The integer storage is copied unchanged from the input tensor, so the
    # underlying representation must match the input and the reference
    # bit-exactly.
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(inp))
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


def _assert_per_channel_float_qparams(
    res_out, ref_out, inp, scales, zero_points, axis, equal_nan=False
):
    # A *floating-point* zero_point tensor switches the output qscheme to
    # torch.per_channel_affine_float_qparams, whose getters report float32
    # scales and zero_points (unlike the int64-zero_point path above).
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == _QUANT_DTYPE[inp.dtype]
    assert res_out.shape == ref_out.shape
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert (
        res_out.qscheme() == ref_out.qscheme() == torch.per_channel_affine_float_qparams
    )
    assert res_out.q_per_channel_axis() == ref_out.q_per_channel_axis() == axis

    assert (
        res_out.q_per_channel_scales().dtype
        == ref_out.q_per_channel_scales().dtype
        == torch.float32
    )
    assert (
        res_out.q_per_channel_zero_points().dtype
        == ref_out.q_per_channel_zero_points().dtype
        == torch.float32
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        ref_out.q_per_channel_scales(),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        ref_out.q_per_channel_zero_points(),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        utils.to_reference(scales).to(torch.float32),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        utils.to_reference(zero_points).to(torch.float32),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(inp))
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_value_ranges(
    shape, axis, value_range, storage_dtype
):
    # The storage tensor and the per-channel metadata are the only value-
    # carrying inputs. Feed all three from the shared value-range table so every
    # numeric range (negative, positive, full dtype bounds, degenerate
    # constants) is exercised; the copy and the metadata must be preserved
    # bit-exactly regardless of sign, magnitude or dtype bounds. The scales use
    # float64 (the canonical quantizer storage dtype), including the +/- extremes
    # of the range table, which the reference stores verbatim.
    num_channels = _num_channels(shape, axis)
    inp = _make_range_tensor(storage_dtype, shape, value_range)
    scales = _make_range_tensor(torch.float64, (num_channels,), value_range)
    zero_points = _make_range_tensor(storage_dtype, (num_channels,), value_range)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor(shape, axis, storage_dtype, scale_dtype):
    # The default overload over a representative value range, covering both
    # input scale dtypes: the quantizer must canonicalize float32 and float64
    # scales to float64 exactly (no lossy round-trip through a wider type).
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(res_out, ref_out, inp, scales, zero_points, axis)
    # The input is only read; it must be untouched.
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("shape,axis", _AXIS_SHAPES)
def test__make_per_channel_quantized_tensor_axis(
    shape, axis, storage_dtype, scale_dtype
):
    # Every valid axis position (positive and negative) across ranks 1-5: the
    # stored q_per_channel_axis must be the caller-supplied value and the
    # metadata/storage must be preserved for each encoding.
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_boundary_values(storage_dtype):
    # Pin the exact storage dtype bounds (make_tensor draws strictly inside the
    # range): min/max/0/(+/-1) must round-trip bit-exactly through the storage
    # copy, and the full-range zero_points must be stored verbatim.
    info = torch.iinfo(storage_dtype)
    values = [info.min, info.max, 0, 1]
    if storage_dtype != torch.uint8:
        values.append(-1)
    num_channels = len(values)
    inp = torch.tensor(
        values * 4, dtype=storage_dtype, device=flag_gems.device
    ).reshape(4, num_channels)
    axis = 1
    scales = torch.linspace(
        0.5, 1.5, num_channels, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.tensor(values, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test__make_per_channel_quantized_tensor_non_contiguous(storage_dtype, scale_dtype):
    # A transposed view whose strides do not match the contiguous layout: the
    # reference materializes/iterates the logical values, so they must be
    # preserved in the output storage regardless of the physical layout.
    base = _make_range_tensor(storage_dtype, (4, 3, 8), ["0", "max"])
    inp = base.transpose(0, 1)
    assert not inp.is_contiguous()  # shape (3, 4, 8)
    axis = 1
    scales, zero_points = _make_metadata(inp.shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_float_zero_points(
    shape, axis, storage_dtype, zero_point_dtype
):
    # A floating-point zero_point tensor selects the
    # per_channel_affine_float_qparams scheme (float32 scales/zero_points
    # stored as-is) instead of the int64-zero_point per_channel_affine scheme.
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_float_metadata(shape, axis, zero_point_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_float_qparams(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("bad", _NON_FINITE)
def test__make_per_channel_quantized_tensor_non_finite_scales(storage_dtype, bad):
    # nan/inf/-inf scales are accepted by both references and stored verbatim.
    # equal_nan is required because the exact-equality helpers compare
    # nan != nan by default.
    shape, axis = (2, 3, 4), 1
    num_channels = _num_channels(shape, axis)
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales = torch.full(
        (num_channels,), bad, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.zeros(
        num_channels, dtype=storage_dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_affine(
        res_out, ref_out, inp, scales, zero_points, axis, equal_nan=True
    )


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("bad", _NON_FINITE)
def test__make_per_channel_quantized_tensor_non_finite_float_zero_points(
    storage_dtype, bad
):
    # Non-finite float zero_points follow the float_qparams path and are stored
    # verbatim (equal_nan handles the nan entry).
    shape, axis = (2, 3, 4), 1
    num_channels = _num_channels(shape, axis)
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales = torch.full(
        (num_channels,), 0.5, dtype=torch.float32, device=flag_gems.device
    )
    zero_points = torch.full(
        (num_channels,), bad, dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_float_qparams(
        res_out, ref_out, inp, scales, zero_points, axis, equal_nan=True
    )


# aten::_make_per_channel_quantized_tensor.out(Tensor self, Tensor scale, Tensor
# zero_point, int axis, *, Tensor(a!) out) -> Tensor(a!) overwrites the
# quantizer metadata and the storage of the provided out tensor (keeping its
# qscheme and dtype) and returns the same object (alias semantics). The out
# buffer must already carry the derived quantized dtype.
@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS)
def test__make_per_channel_quantized_tensor_out(shape, axis, storage_dtype):
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out_buf = _make_out_buffer(
        shape, axis, storage_dtype, _ref_device(), reference=True
    )
    ref_ret = torch.ops.aten._make_per_channel_quantized_tensor.out(
        ref_inp, ref_scales, ref_zero_points, axis, out=ref_out_buf
    )
    assert ref_ret is ref_out_buf

    act_out_buf = _make_out_buffer(shape, axis, storage_dtype, flag_gems.device)
    res_ret = _resolve("_make_per_channel_quantized_tensor.out")(
        inp, scales, zero_points, axis, out=act_out_buf
    )
    assert res_ret is act_out_buf

    _assert_per_channel_affine(act_out_buf, ref_out_buf, inp, scales, zero_points, axis)
    utils.gems_assert_equal(inp, ref_inp)


# ---------------------------------------------------------------------------
# Negative cases: each invalid request must raise on the aten reference and the
# candidate must reject it too rather than silently succeeding.
# ---------------------------------------------------------------------------


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("invalid_dtype", _REJECTED_INPUT_DTYPES)
def test__make_per_channel_quantized_tensor_rejects_non_storage_dtype(invalid_dtype):
    # Only integer storage dtypes (uint8/int8/int32) can be reinterpreted as
    # quantized storage; float and other integer dtypes are rejected.
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=invalid_dtype, device=flag_gems.device)
    scales = torch.full((3,), 0.5, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(3, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_dtype", [torch.int32, torch.int64])
def test__make_per_channel_quantized_tensor_rejects_non_float_scales(scale_dtype):
    # The scale tensor must be floating point; integer scales are rejected.
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.tensor([1, 2, 3], dtype=scale_dtype, device=flag_gems.device)
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("bad_metadata", ["scale", "zero_point"])
def test__make_per_channel_quantized_tensor_rejects_non_1d_metadata(bad_metadata):
    # The per-channel metadata must be 1-D (one entry per channel).
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(3, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(3, dtype=torch.int64, device=flag_gems.device)
    if bad_metadata == "scale":
        scales = torch.rand(2, 3, dtype=torch.float32, device=flag_gems.device)
    else:
        zero_points = torch.zeros(2, 3, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_len,zero_point_len", [(2, 3), (3, 2), (0, 3), (3, 0)])
def test__make_per_channel_quantized_tensor_rejects_metadata_length_mismatch(
    scale_len, zero_point_len
):
    # The factory requires scales.numel() == zero_points.numel(); the lengths
    # only need to match each other (they are not checked against size[axis]).
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(scale_len, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(
        zero_point_len, dtype=torch.int64, device=flag_gems.device
    )

    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out_rejects_non_quantized_buffer(
    storage_dtype,
):
    # The .out overload keeps the out tensor dtype, which must already be the
    # derived quantized dtype; a plain (non-quantized) buffer is rejected.
    shape, axis = (2, 3), 1
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = utils.to_reference(inp)

    ref_buf = torch.empty(shape, dtype=torch.float32, device=_ref_device())
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor.out(
            ref_inp,
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            axis,
            out=ref_buf,
        )

    act_buf = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor.out")(
            inp, scales, zero_points, axis, out=act_buf
        )


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out_rejects_wrong_quantized_dtype(
    storage_dtype,
):
    # A quantized buffer of any other dtype (e.g. qint8 for a quint8 output) is
    # rejected as well.
    shape, axis = (2, 3), 1
    num_channels = _num_channels(shape, axis)
    inp = _make_range_tensor(storage_dtype, shape, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = utils.to_reference(inp)

    ref_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(
            torch.full(
                (num_channels,),
                9.0,
                dtype=torch.float64,
                device=flag_gems.device,
            )
        ),
        zero_points=utils.to_reference(
            torch.full((num_channels,), 9, dtype=torch.int64, device=flag_gems.device)
        ),
        axis=axis,
        dtype=_WRONG_QUANT_DTYPE[storage_dtype],
        device=_ref_device(),
    )
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor.out(
            ref_inp,
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            axis,
            out=ref_buf,
        )

    act_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full(
            (num_channels,), 9.0, dtype=torch.float64, device=flag_gems.device
        ),
        zero_points=torch.full(
            (num_channels,), 9, dtype=torch.int64, device=flag_gems.device
        ),
        axis=axis,
        dtype=_WRONG_QUANT_DTYPE[storage_dtype],
        device=flag_gems.device,
    )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_make_per_channel_quantized_tensor.out")(
            inp, scales, zero_points, axis, out=act_buf
        )
