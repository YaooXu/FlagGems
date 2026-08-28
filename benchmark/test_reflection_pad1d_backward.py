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

# (batch, width) pairs covering small to medium tensors for pad1d backward
REFLECTION_PAD1D_BACKWARD_SHAPES = [
    (2, 3),
    (4, 8),
    (8, 16),
    (1, 32),
    (4, 64),
    (8, 128),
]


class ReflectionPad1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REFLECTION_PAD1D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) == 2:
                B, W = shape
            else:
                B, W = 1, shape[0]
            padding = (1, 2)
            x = torch.randn(B, W, dtype=cur_dtype, device=self.device)
            # Compute forward to get output size
            padded = torch.ops.aten.reflection_pad1d(x, padding)
            W_out = padded.shape[-1]
            grad = torch.ones(B, W_out, dtype=cur_dtype, device=self.device)
            yield grad, x, padding

    def get_case_iter(self, dtype):
        padding = (1, 2)
        for ordinal, shape in enumerate(self.shapes):
            if len(shape) == 2:
                B, W = shape
            else:
                B, W = 1, shape[0]
            W_out = W + padding[0] + padding[1]
            grad_shape = (B, W_out)
            inp_shape = (B, W)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_output": grad_shape, "input": inp_shape},
                    params={"padding": padding},
                    builder_args=(inp_shape, padding),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        inp_shape, padding = plan.builder_args
        B, W = inp_shape
        W_out = W + padding[0] + padding[1]
        x = torch.randn(inp_shape, dtype=case.dtype, device=self.device)
        grad = torch.ones((B, W_out), dtype=case.dtype, device=self.device)
        return grad, x, padding


@pytest.mark.reflection_pad1d_backward
def test_reflection_pad1d_backward():
    bench = ReflectionPad1dBackwardBenchmark(
        op_name="reflection_pad1d_backward",
        torch_op=torch.ops.aten.reflection_pad1d_backward,
        gems_op=flag_gems.reflection_pad1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
