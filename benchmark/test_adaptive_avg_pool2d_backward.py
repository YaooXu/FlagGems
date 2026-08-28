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


class AdaptiveAvgPool2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Common CNN feature-map shapes paired with representative output sizes.
        self.shapes = [
            (4, 3, 32, 32, 7, 7),
            (8, 64, 56, 56, 7, 7),
            (4, 128, 112, 112, 14, 14),
        ]
        self.shape_desc = "N, C, H, W, OUT_H, OUT_W"

    def get_case_iter(self, dtype):
        for ordinal, (n, c, h, w, out_h, out_w) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"inp": [n, c, h, w], "grad_output": [n, c, out_h, out_w]},
                    params={},
                    builder_args=(n, c, h, w, out_h, out_w),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, c, h, w, out_h, out_w = plan.builder_args
        inp = torch.randn((n, c, h, w), dtype=case.dtype, device=self.device)
        grad_output = torch.randn(
            (n, c, out_h, out_w), dtype=case.dtype, device=self.device
        )
        return grad_output, inp


@pytest.mark.adaptive_avg_pool2d_backward
def test_adaptive_avg_pool2d_backward():
    bench = AdaptiveAvgPool2dBackwardBenchmark(
        op_name="adaptive_avg_pool2d_backward",
        torch_op=torch.ops.aten._adaptive_avg_pool2d_backward,
        gems_op=flag_gems._adaptive_avg_pool2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
