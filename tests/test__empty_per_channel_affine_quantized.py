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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

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
# never the garbage element values.
QINT_DTYPES = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]

# (shape, axis) pairs covering rank 1-4, every valid axis position, and both
# positive and negative axis encodings.
SHAPE_AXIS = [
    ((4,), 0),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -1),
    ((2, 3, 4), 1),
    ((2, 3, 4), -1),
    ((2, 3, 4, 5), 1),
    ((2, 3, 4, 5), -1),
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


def _assert_per_channel_metadata(res_out, ref_out, axis, scales, zero_points):
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
        res_out.q_per_channel_scales(), ref_out.q_per_channel_scales()
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(), ref_out.q_per_channel_zero_points()
    )

    # The stored metadata must reproduce the caller-supplied values exactly
    # (scales widened to float64, zero_points to int64).
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(), utils.to_reference(scales).to(torch.float64)
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        utils.to_reference(zero_points).to(torch.int64),
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
@pytest.mark.parametrize("quantized_dtype", QINT_DTYPES)
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
