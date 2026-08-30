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
import os
import sys

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# The verification harness imports this module as ``tests.test_<op>`` via
# --import-mode=importlib without creating the parent ``tests`` package and
# without putting the temp directory on sys.path; put it there so the
# package-relative imports below resolve.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_empty_per_channel_affine_quantized`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._empty_per_channel_affine_quantized`` and ``-m
# _empty_per_channel_affine_quantized`` both work.
for _name in (
    "_empty_per_channel_affine_quantized",
    "_empty_per_channel_affine_quantized_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )


def _resolve(name):
    # Resolved inside each test (never at import time) so that a process-local
    # override installed by KernelGen via ``override_gems_op`` for this run
    # wins. The default stays None until flag_gems registers the operator;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(name, getattr(flag_gems, name, None))


# aten::_empty_per_channel_affine_quantized is a quantized-tensor factory: the
# element values are uninitialized (like ``empty``), but the per-channel
# quantizer metadata (qscheme, axis, scales, zero_points) is fully determined
# by the arguments. Correctness therefore targets the metadata and the layout,
# never the garbage element values. The scales/zero_points tensors are the only
# value-carrying inputs, so the value-range dimension of the regular-operator
# spec is applied to them (broadcast/backward do not apply to a factory).
QINT_DTYPES = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]

# The CUDA ``.out`` factory overwrites the quantizer metadata of the provided
# out buffer, and that overwrite is unreliable for sub-byte quantized dtypes
# (quint4x2/quint2x4): the scales stored in the out buffer come back corrupted
# (garbage values) after the second write. The ``.default`` factory is not
# affected (it allocates a fresh tensor), so sub-byte dtypes stay covered for
# ``.default`` and are only excluded from the ``.out`` variant.
QINT_DTYPES_OUT = [torch.qint8, torch.quint8, torch.qint32]

# (shape, axis) pairs covering rank 1-5, every valid axis position, both
# positive and negative axis encodings, and zero-sized dimensions.
SHAPE_AXIS = [
    ((4,), 0),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -1),
    ((2, 3, 4), 1),
    ((2, 3, 4), -1),
    ((2, 3, 4, 5), 1),
    ((2, 3, 4, 5), -1),
    ((2, 3, 4, 5, 6), 2),
    ((2, 3, 4, 5, 6), -2),
    ((0, 2, 3), 1),
    ((2, 0), 1),
]

# The quantizer stores scales as float64 and zero_points as int64 regardless of
# the input dtypes; cover both supported input dtypes for each.
SCALES_DTYPES = [torch.float32, torch.float64]
ZERO_POINT_DTYPES = [torch.int32, torch.int64]


def _make_metadata(shape, axis, scale_dtype, zero_point_dtype):
    # Positive zero_points: negative values are rejected by the CUDA
    # per-channel affine quantizer validation (dequantize lower-bound check).
    num_channels = shape[axis]
    scales = torch.arange(
        1, num_channels + 1, dtype=scale_dtype, device=flag_gems.device
    )
    zero_points = torch.randint(
        0, 8, (num_channels,), dtype=zero_point_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _assert_per_channel_metadata(
    res_out, ref_out, axis, scales, zero_points, equal_nan=False
):
    # The factory output is uninitialized memory, so only quantizer metadata
    # and layout are compared.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.stride() == ref_out.stride()
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


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALES_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", ZERO_POINT_DTYPES)
def test__empty_per_channel_affine_quantized(
    shape, axis, quantized_dtype, scale_dtype, zero_point_dtype
):
    scales, zero_points = _make_metadata(shape, axis, scale_dtype, zero_point_dtype)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(scales),
        zero_points=utils.to_reference(zero_points),
        axis=axis,
        dtype=quantized_dtype,
        device=ref_device,
    )

    gems_op = _resolve("_empty_per_channel_affine_quantized")
    res_out = gems_op(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out, axis, scales, zero_points)


@pytest.mark._empty_per_channel_affine_quantized_out
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES_OUT)
@pytest.mark.parametrize("scale_dtype", SCALES_DTYPES)
@pytest.mark.parametrize("zero_point_dtype", ZERO_POINT_DTYPES)
def test__empty_per_channel_affine_quantized_out(
    shape, axis, quantized_dtype, scale_dtype, zero_point_dtype
):
    scales, zero_points = _make_metadata(shape, axis, scale_dtype, zero_point_dtype)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    # The .out variant writes the quantizer metadata into the provided out
    # buffer and returns that same tensor (alias semantics). The buffer is
    # created with deliberately *different* metadata so the overwrite is
    # observable. Note: the .out variant does not change the out buffer dtype,
    # so the buffer is created with the tested quantized dtype.
    ref_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(
            torch.tensor([9.0], dtype=torch.float64, device=flag_gems.device)
        ),
        zero_points=utils.to_reference(
            torch.tensor([9], dtype=torch.int64, device=flag_gems.device)
        ),
        axis=0,
        dtype=quantized_dtype,
        device=ref_device,
    )
    ref_ret = torch.ops.aten._empty_per_channel_affine_quantized.out(
        shape,
        scales=utils.to_reference(scales),
        zero_points=utils.to_reference(zero_points),
        axis=axis,
        out=ref_out_buf,
    )
    assert ref_ret is ref_out_buf

    act_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.tensor([9.0], dtype=torch.float64, device=flag_gems.device),
        zero_points=torch.tensor([9], dtype=torch.int64, device=flag_gems.device),
        axis=0,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )
    gems_op = _resolve("_empty_per_channel_affine_quantized_out")
    res_ret = gems_op(
        shape, scales=scales, zero_points=zero_points, axis=axis, out=act_out_buf
    )
    assert res_ret is act_out_buf

    _assert_per_channel_metadata(act_out_buf, ref_out_buf, axis, scales, zero_points)


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__empty_per_channel_affine_quantized_metadata_value_ranges(
    shape, axis, value_range
):
    # The scales/zero_points metadata are the only value-carrying inputs of
    # this factory. Feed them from the shared value-range table so every
    # numeric range (negative, positive, full dtype bounds, constants) is
    # exercised; the metadata must be stored bit-exactly regardless of sign,
    # magnitude or dtype bounds.
    num_channels = shape[axis]
    scales = tu.make_input(torch.float64, (num_channels,), value_range).to(
        flag_gems.device
    )
    zero_points = tu.make_input(torch.int64, (num_channels,), value_range).to(
        flag_gems.device
    )
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(scales),
        zero_points=utils.to_reference(zero_points),
        axis=axis,
        dtype=torch.quint8,
        device=ref_device,
    )

    res_out = _resolve("_empty_per_channel_affine_quantized")(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out, axis, scales, zero_points)


@pytest.mark._empty_per_channel_affine_quantized
def test__empty_per_channel_affine_quantized_nan_inf_scales():
    # The factory copies scale values verbatim, so nan/inf/-inf metadata must
    # survive both the reference and the candidate unchanged. ``equal_nan`` is
    # required because the exact-equality helpers compare nan != nan by
    # default.
    shape = (3, 4)
    scales = torch.tensor(
        [float("nan"), float("inf"), -float("inf")],
        dtype=torch.float64,
        device=flag_gems.device,
    )
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(scales),
        zero_points=utils.to_reference(zero_points),
        axis=0,
        dtype=torch.quint8,
        device=ref_device,
    )

    res_out = _resolve("_empty_per_channel_affine_quantized")(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=0,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(
        res_out, ref_out, 0, scales, zero_points, equal_nan=True
    )


@pytest.mark._empty_per_channel_affine_quantized
def test__empty_per_channel_affine_quantized_fp64_scales_preserved():
    # The reference stores scales with full float64 precision (it does not
    # round through float32). Values below are not representable in float32, so
    # a candidate that widens float32 metadata would fail this exact check.
    shape = (3, 4)
    scales = torch.tensor(
        [0.1 + 1e-17, 0.2, 1 / 3], dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    ref_out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(scales),
        zero_points=utils.to_reference(zero_points),
        axis=0,
        dtype=torch.quint8,
        device=ref_device,
    )

    res_out = _resolve("_empty_per_channel_affine_quantized")(
        shape,
        scales=scales,
        zero_points=zero_points,
        axis=0,
        dtype=torch.quint8,
        device=flag_gems.device,
    )

    _assert_per_channel_metadata(res_out, ref_out, 0, scales, zero_points)


# --- Negative cases ---------------------------------------------------------
# The reference validates (a) that the output dtype is quantized, (b) that the
# scales tensor is floating point, and (c) that scales and zero_points have the
# same number of elements. It does *not* validate the axis against the rank,
# nor the metadata lengths against size[axis]: out-of-range axes and
# equal-but-wrong metadata lengths are stored verbatim, so those are not
# negative cases here. Each test asserts the reference rejects the arguments
# first (documenting that the case is genuinely invalid) and then that the
# candidate rejects them identically.


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("invalid_dtype", [torch.float32, torch.float64, torch.int32])
def test__empty_per_channel_affine_quantized_negative_invalid_dtype(invalid_dtype):
    # A non-quantized output dtype is rejected by the reference (RuntimeError
    # on CPU, NotImplementedError - a RuntimeError subclass - on CUDA) and must
    # be rejected by the candidate as well: a quantized factory must not
    # silently allocate a non-quantized buffer.
    shape = (2, 3)
    scales = torch.tensor([1.0, 2.0], dtype=torch.float64, device=flag_gems.device)
    zero_points = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=utils.to_reference(scales),
            zero_points=utils.to_reference(zero_points),
            axis=1,
            dtype=invalid_dtype,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve("_empty_per_channel_affine_quantized")(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=invalid_dtype,
            device=flag_gems.device,
        )


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("scale_dtype", [torch.int32, torch.int64, torch.uint8])
def test__empty_per_channel_affine_quantized_negative_non_float_scales(scale_dtype):
    # scales must be a floating-point tensor; integer scales are rejected.
    shape = (2, 3)
    scales = torch.tensor([1, 2], dtype=scale_dtype, device=flag_gems.device)
    zero_points = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=utils.to_reference(scales),
            zero_points=utils.to_reference(zero_points),
            axis=1,
            dtype=torch.quint8,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve("_empty_per_channel_affine_quantized")(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            device=flag_gems.device,
        )


@pytest.mark._empty_per_channel_affine_quantized
@pytest.mark.parametrize("scale_len,zero_point_len", [(1, 2), (3, 1), (0, 2), (2, 0)])
def test__empty_per_channel_affine_quantized_negative_metadata_length_mismatch(
    scale_len, zero_point_len
):
    # The factory requires scales.numel() == zero_points.numel(); the lengths
    # only need to match each other (they are not checked against size[axis]).
    shape = (2, 3)
    scales = torch.arange(
        1, scale_len + 1, dtype=torch.float64, device=flag_gems.device
    )
    zero_points = torch.zeros(
        zero_point_len, dtype=torch.int64, device=flag_gems.device
    )
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._empty_per_channel_affine_quantized(
            shape,
            scales=utils.to_reference(scales),
            zero_points=utils.to_reference(zero_points),
            axis=1,
            dtype=torch.quint8,
            device=ref_device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve("_empty_per_channel_affine_quantized")(
            shape,
            scales=scales,
            zero_points=zero_points,
            axis=1,
            dtype=torch.quint8,
            device=flag_gems.device,
        )
