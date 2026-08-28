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


def cudnn_convolution_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (
        inp,
        weight,
        [padding, padding],
        [stride, stride],
        [1, 1],
        groups,
        False,
        False,
        False,
    )


class CudnnConv2dBenchmark(base.GenericBenchmark):
    SHAPES = [
        (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
        (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
        (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
    ]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.SHAPES:
            yield from self.input_fn(shape, dtype, self.device)

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, shape in enumerate(self.SHAPES):
            (
                batch,
                input_c,
                input_h,
                input_w,
                out_c,
                kernel_h,
                kernel_w,
                stride,
                padding,
                groups,
            ) = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "input": (batch, input_c, input_h, input_w),
                        "weight": (
                            out_c,
                            input_c // groups,
                            kernel_h,
                            kernel_w,
                        ),
                    },
                    params={
                        "padding": [padding, padding],
                        "stride": [stride, stride],
                        "dilation": [1, 1],
                        "groups": groups,
                        "benchmark": False,
                        "deterministic": False,
                        "allow_tf32": False,
                    },
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape, _ = case.builder_args[0].builder_args
        return next(cudnn_convolution_input_fn(shape, case.dtype, self.device))


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    bench = CudnnConv2dBenchmark(
        input_fn=cudnn_convolution_input_fn,
        op_name="cudnn_convolution",
        torch_op=torch.ops.aten.cudnn_convolution.default,
        gems_op=flag_gems.cudnn_convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
