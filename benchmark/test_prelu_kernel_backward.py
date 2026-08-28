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

import flag_gems

from . import base, consts

# Shapes for prelu_kernel_backward benchmark
PRELU_KERNEL_BACKWARD_SHAPES = [
    (16, 128, 64, 1280),  # Large 4D shape
    (1024, 1024),  # 2D
    (16, 7, 57, 32, 29),  # 5D
]


class PReluKernelBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PRELU_KERNEL_BACKWARD_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_output": shape, "x": shape, "weight": (1,)},
                    params={},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        grad_output = torch.randn(shape, dtype=case.dtype, device=self.device)
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        weight = torch.tensor([0.25], dtype=case.dtype, device=self.device)
        return grad_output, x, weight


@pytest.mark.prelu_kernel_backward
def test_prelu_kernel_backward():
    bench = PReluKernelBackwardBenchmark(
        op_name="prelu_kernel_backward",
        torch_op=torch.ops.aten._prelu_kernel_backward,
        gems_op=flag_gems._prelu_kernel_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
