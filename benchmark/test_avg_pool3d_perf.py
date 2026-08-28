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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def avg_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # Common case
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With count_include_pad=False
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
            "divisor_override": None,
        }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": True,
            "divisor_override": None,
        }
        # With divisor_override
        if shape[-3] >= 2 and shape[-2] >= 2 and shape[-1] >= 2:
            yield inp, {
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3,
            }


class AvgPool3dBenchmark(base.GenericBenchmark):
    SHAPES_5D = [
        (4, 3, 16, 56, 56),
        (8, 64, 8, 28, 28),
        (16, 128, 4, 14, 14),
        (32, 256, 4, 7, 7),
    ]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.SHAPES_5D:
            yield from self.input_fn(shape, dtype, self.device)

    def get_case_iter(self, dtype) -> Generator:
        ordinal = 0
        for shape in self.SHAPES_5D:
            configs = [
                {
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "ceil_mode": False,
                    "count_include_pad": True,
                    "divisor_override": None,
                }
            ]
            if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
                configs.extend(
                    [
                        {**configs[0], "count_include_pad": False},
                        {**configs[0], "ceil_mode": True},
                    ]
                )
                if min(shape[-3:]) >= 2:
                    configs.append(
                        {
                            "kernel_size": 2,
                            "stride": 1,
                            "padding": 0,
                            "ceil_mode": False,
                            "count_include_pad": True,
                            "divisor_override": 3,
                        }
                    )
            for config in configs:
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": shape},
                        params=config,
                        builder_args=(shape, config),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        shape, config = case.builder_args[0].builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        return inp, config


class AvgPool3dBackwardBenchmark(AvgPool3dBenchmark):
    def build_inputs(self, case):
        shape, config = case.builder_args[0].builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        out = torch.ops.aten.avg_pool3d(inp, **config)
        grad_output = torch.randn_like(out)
        return grad_output, inp, config


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.avg_pool3d_backward
def test_perf_avg_pool3d_backward():
    bench = AvgPool3dBackwardBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d_backward",
        torch_op=torch.ops.aten.avg_pool3d_backward,
        gems_op=flag_gems.avg_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
