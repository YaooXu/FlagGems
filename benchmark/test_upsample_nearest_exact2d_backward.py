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


class UpsampleNearestExact2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Typical feature map sizes: small, medium, large spatial dims
        self.shapes = [(2, 3, 8, 8), (4, 8, 16, 16), (8, 16, 32, 32)]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Create grad_output by doing a forward pass first
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            out_h = shape[2] * 2
            out_w = shape[3] * 2
            output_size = (out_h, out_w)

            # Forward pass to get output
            out = torch.ops.aten._upsample_nearest_exact2d(
                x, [out_h, out_w], None, None
            )
            grad_output = torch.ones_like(out)

            input_size = tuple(x.shape)
            yield grad_output, output_size, input_size

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            output_size = (shape[2] * 2, shape[3] * 2)
            grad_shape = shape[:2] + output_size
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"grad_output": grad_shape, "input": shape},
                    params={"output_size": output_size, "input_size": shape},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        out_h = shape[2] * 2
        out_w = shape[3] * 2
        output_size = (out_h, out_w)
        out = torch.ops.aten._upsample_nearest_exact2d(
            x, [out_h, out_w], None, None
        )
        grad_output = torch.ones_like(out)
        input_size = tuple(x.shape)
        return grad_output, output_size, input_size


@pytest.mark.upsample_nearest_exact2d_backward
def test_upsample_nearest_exact2d_backward():
    bench = UpsampleNearestExact2dBackwardBenchmark(
        op_name="upsample_nearest_exact2d_backward",
        torch_op=torch.ops.aten._upsample_nearest_exact2d_backward.default,
        gems_op=flag_gems._upsample_nearest_exact2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
