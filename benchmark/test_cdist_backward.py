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

from . import base


class CdistBackwardBenchmark(base.Benchmark):
    def set_more_shapes(self):
        return [
            (2, 16, 32),
            (4, 32, 64),
            (8, 64, 128),
            (16, 128, 256),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, n1, dim = shape
            n2 = n1 // 2 + 1
            x1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(batch, n2, dim, dtype=cur_dtype, device=self.device)
            cdist = torch.cdist(x1, x2, p=2.0)
            grad = torch.randn(batch, n1, n2, dtype=cur_dtype, device=self.device)
            yield grad, x1, x2, 2.0, cdist

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            batch, n1, dim = shape
            n2 = n1 // 2 + 1
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "grad": (batch, n1, n2),
                        "x1": shape,
                        "x2": (batch, n2, dim),
                        "cdist": (batch, n1, n2),
                    },
                    params={"p": 2.0},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        batch, n1, dim = shape
        n2 = n1 // 2 + 1
        x1 = torch.randn(shape, dtype=case.dtype, device=self.device)
        x2 = torch.randn(batch, n2, dim, dtype=case.dtype, device=self.device)
        cdist = torch.cdist(x1, x2, p=2.0)
        grad = torch.randn(batch, n1, n2, dtype=case.dtype, device=self.device)
        return grad, x1, x2, 2.0, cdist


@pytest.mark.cdist_backward
def test_cdist_backward():
    bench = CdistBackwardBenchmark(
        op_name="cdist_backward",
        torch_op=torch.ops.aten._cdist_backward,
        gems_op=flag_gems._cdist_backward,
        # _cdist_backward uses fp32 accumulation; only float32 is numerically stable
        dtypes=[torch.float32],
    )
    bench.run()
