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


def _resolve(name):
    # Resolved inside each test (never at import time) so that a process-local
    # override installed by KernelGen via ``override_gems_op`` for this run
    # wins. The default stays None until flag_gems registers the operator;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(name, getattr(flag_gems, name, None))


# aten::_make_per_channel_quantized_tensor(self, scale, zero_point, axis)
# reinterprets a *plain integer* tensor (uint8/int8/int32) as the underlying
# storage of a torch.per_channel_affine quantized tensor carrying the given
# per-channel scale/zero_point metadata. The output quantized dtype is derived
# from the input storage dtype: uint8 -> quint8, int8 -> qint8, int32 ->
# qint32; the integer values are copied to the output storage unchanged.
STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]

# (shape, axis) pairs covering rank 1-5, every valid axis position, and both
# positive and negative axis encodings. num_channels = shape[axis] varies so
# the per-channel metadata length is exercised.
SHAPE_AXIS = (
    [((2, 3, 4), 1)]
    if utils.QUICK_MODE
    else [
        ((7,), 0),
        ((7,), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((2, 3, 4), 1),
        ((2, 3, 4), -2),
        ((2, 3, 4, 5), 2),
        ((2, 3, 4, 5), -1),
        ((16, 32, 64), 1),
    ]
)

# The quantizer stores scales as float64 regardless of the input dtype; cover
# both supported scale dtypes.
SCALE_DTYPES = [torch.float32, torch.float64]


def _make_input(shape, storage_dtype):
    if storage_dtype == torch.uint8:
        return torch.randint(
            0, 256, shape, dtype=storage_dtype, device=flag_gems.device
        )
    if storage_dtype == torch.int8:
        return torch.randint(
            -128, 128, shape, dtype=storage_dtype, device=flag_gems.device
        )
    info = torch.iinfo(torch.int32)
    return torch.randint(
        info.min, info.max, shape, dtype=storage_dtype, device=flag_gems.device
    )


def _make_metadata(shape, axis, storage_dtype, scale_dtype):
    # Positive scales, and zero_points kept inside the valid range of each
    # storage dtype ([1, 128) is valid for uint8/int8/int32), so the
    # per-channel affine quantizer validation always accepts them.
    num_channels = shape[axis]
    scales = torch.rand(num_channels, dtype=scale_dtype, device=flag_gems.device) + 0.1
    zero_points = torch.randint(
        1, 128, (num_channels,), dtype=storage_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis):
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
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

    # The integer storage is copied unchanged from the input tensor, so the
    # underlying representation must match the input and the reference
    # bit-exactly.
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(inp))
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test__make_per_channel_quantized_tensor(shape, axis, storage_dtype, scale_dtype):
    inp = _make_input(shape, storage_dtype)
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    gems_op = _resolve("_make_per_channel_quantized_tensor")
    res_out = gems_op(inp, scales, zero_points, axis)

    _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test__make_per_channel_quantized_tensor_non_contiguous(storage_dtype, scale_dtype):
    # A transposed view whose strides do not match the contiguous layout: the
    # reference CPU kernel materializes a contiguous copy and the CUDA kernel
    # iterates over the input strides, so the logical (not physical) values
    # must be preserved in the output storage.
    inp = _make_input((4, 3, 8), storage_dtype).transpose(0, 1)  # shape (3, 4, 8)
    assert not inp.is_contiguous()
    axis = 1
    scales, zero_points = _make_metadata(inp.shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    gems_op = _resolve("_make_per_channel_quantized_tensor")
    res_out = gems_op(inp, scales, zero_points, axis)

    _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out(shape, axis, storage_dtype):
    inp = _make_input(shape, storage_dtype)
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device

    # The .out variant writes the quantizer metadata and the storage into the
    # provided out buffer and returns that same tensor (alias semantics). The
    # buffer is created with deliberately *different* metadata so the
    # overwrite is observable. The .out variant keeps the out buffer dtype,
    # which must be the quantized dtype derived from the storage dtype.
    quantized_dtype = {
        torch.uint8: torch.quint8,
        torch.int8: torch.qint8,
        torch.int32: torch.qint32,
    }[storage_dtype]

    ref_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(
            torch.full(
                (shape[axis],), 9.0, dtype=torch.float64, device=flag_gems.device
            )
        ),
        zero_points=utils.to_reference(
            torch.full((shape[axis],), 9, dtype=torch.int64, device=flag_gems.device)
        ),
        axis=axis,
        dtype=quantized_dtype,
        device=ref_device,
    )
    ref_ret = torch.ops.aten._make_per_channel_quantized_tensor.out(
        ref_inp, ref_scales, ref_zero_points, axis, out=ref_out_buf
    )
    assert ref_ret is ref_out_buf

    act_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full(
            (shape[axis],), 9.0, dtype=torch.float64, device=flag_gems.device
        ),
        zero_points=torch.full(
            (shape[axis],), 9, dtype=torch.int64, device=flag_gems.device
        ),
        axis=axis,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )
    gems_op = _resolve("_make_per_channel_quantized_tensor_out")
    res_ret = gems_op(inp, scales, zero_points, axis, out=act_out_buf)
    assert res_ret is act_out_buf

    _assert_per_channel_quantized(
        act_out_buf, ref_out_buf, inp, scales, zero_points, axis
    )
