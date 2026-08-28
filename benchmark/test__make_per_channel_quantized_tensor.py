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

from . import base

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

# aten::_make_per_channel_quantized_tensor copies a plain integer storage
# tensor (uint8/int8/int32) into a per-channel affine quantized tensor. There
# is no core_shapes.yaml entry for it, so the base class would fall back to
# consts.DEFAULT_SHAPES, which includes a 1-B-element 1-D tensor whose
# allocation cost would dominate the measurement. Use a modest,
# allocation-friendly shape set that still spans a realistic range of ranks and
# tensor sizes.
QUANT_STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]

MPCQT_SHAPES = [
    (1024,),
    (4096, 256),
    (1024, 1024),
    (64, 512, 512),
    (16, 128, 64, 1280),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"axis": 0},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    axis = plan.params["axis"]
    num_channels = shape[axis]
    # The storage tensor holds raw integer values; scales stay positive and
    # zero_points stay in [1, 128) so every storage dtype stays in range.
    inp = torch.randint(0, 100, shape, dtype=dtype, device=device)
    scale = torch.rand(num_channels, dtype=torch.float32, device=device) + 0.1
    zero_point = torch.randint(1, 128, (num_channels,), dtype=dtype, device=device)
    return inp, scale, zero_point, axis


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    axis = plan.params["axis"]
    num_channels = shape[axis]
    quantized_dtype = {
        torch.uint8: torch.quint8,
        torch.int8: torch.qint8,
        torch.int32: torch.qint32,
    }[dtype]
    inp = torch.randint(0, 100, shape, dtype=dtype, device=device)
    scale = torch.rand(num_channels, dtype=torch.float32, device=device) + 0.1
    zero_point = torch.randint(1, 128, (num_channels,), dtype=dtype, device=device)
    # The .out variant writes into (and returns) the provided buffer without
    # changing its dtype, so the buffer is created with the quantized dtype
    # derived from the benchmarked storage dtype. The initial metadata is
    # deliberately different so the overwrite performed by the op is observable.
    out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full((num_channels,), 1.0, dtype=torch.float64, device=device),
        zero_points=torch.full((num_channels,), 0, dtype=torch.int64, device=device),
        axis=axis,
        dtype=quantized_dtype,
        device=device,
    )
    return inp, scale, zero_point, axis, {"out": out}


class MakePerChannelQuantizedTensorBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = MPCQT_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark._make_per_channel_quantized_tensor
def test__make_per_channel_quantized_tensor():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="_make_per_channel_quantized_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor,
        gems_op=getattr(flag_gems, "_make_per_channel_quantized_tensor", None),
        dtypes=QUANT_STORAGE_DTYPES,
    )
    bench.run()


@pytest.mark._make_per_channel_quantized_tensor_out
def test__make_per_channel_quantized_tensor_out():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="_make_per_channel_quantized_tensor_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor.out,
        gems_op=getattr(flag_gems, "_make_per_channel_quantized_tensor_out", None),
        dtypes=QUANT_STORAGE_DTYPES,
    )
    bench.run()
