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
from . import conftest as cfg

# ``_empty_affine_quantized`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# markers directly on the MarkGenerator so ``@pytest.mark._empty_affine_quantized``
# and ``-m _empty_affine_quantized`` both work.
for _name in ("_empty_affine_quantized", "_empty_affine_quantized_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Per-tensor affine quantized storage dtypes accepted by the factory.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Representative (dtype-valid) quantization parameters. scale must be positive
# and zero_point must fit every storage dtype used above.
QUANT_SCALES = [0.01, 0.5, 1.0]
QUANT_ZERO_POINTS = [-1, 2]

# Rank-4 shapes: torch.channels_last requires exactly 4 dimensions.
CHANNELS_LAST_SHAPES = [(1, 3, 8, 8), (2, 3, 16, 16), (16, 3, 32, 32)]


def _resolve(name):
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._empty_affine_quantized is registered; resolution order is:
    # (1) override, (2) the direct flag_gems callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(name, getattr(flag_gems, name, None))


def _assert_quant_metadata(res, ref):
    # _empty_affine_quantized is an ``empty``-style factory: the storage bytes
    # are uninitialized, so element values are not part of the contract. The
    # observable contract is purely structural and must match the aten
    # reference: quantization parameters, shape, layout, dtype and device.
    assert res.is_quantized
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape
    assert res.numel() == ref.numel()
    assert res.stride() == ref.stride()
    assert res.q_scale() == ref.q_scale()
    assert res.q_zero_point() == ref.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res.device.type == torch.device(flag_gems.device).type


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized(shape, dtype, scale, zero_point):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=ref_device, scale=scale, zero_point=zero_point
    )

    gems_op = _resolve("_empty_affine_quantized")
    res_out = gems_op(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=zero_point
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous()


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", CHANNELS_LAST_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_channels_last(shape, dtype):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=dtype,
        device=ref_device,
        scale=0.5,
        zero_point=1,
        memory_format=torch.channels_last,
    )

    gems_op = _resolve("_empty_affine_quantized")
    res_out = gems_op(
        shape,
        dtype=dtype,
        device=flag_gems.device,
        scale=0.5,
        zero_point=1,
        memory_format=torch.channels_last,
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous(memory_format=torch.channels_last)


# aten::_empty_affine_quantized.out resets the quantization parameters of the
# provided ``out`` tensor (keeping its shape and dtype) and returns the same
# object (alias semantics).
@pytest.mark._empty_affine_quantized_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized_out(shape, dtype, scale, zero_point):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    # The out buffers start with different quantization parameters so the
    # overwrite performed by the op is observable.
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=ref_device, scale=1.0, zero_point=0
    )
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        shape, scale=scale, zero_point=zero_point, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=1.0, zero_point=0
    )
    gems_op = _resolve("_empty_affine_quantized.out")
    res_out = gems_op(shape, scale=scale, zero_point=zero_point, out=act_out_buf)
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out)
