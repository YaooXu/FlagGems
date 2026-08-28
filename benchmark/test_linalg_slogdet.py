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

# Use linalg-specific square-matrix shapes: one batched small case covers
# the (*, n, n) interface, and 4x4 through 32x32 covers the small/medium
# matrices targeted by this single-program LU implementation.
SLOGDET_SHAPES = [
    (2, 3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        A = torch.randn(shape, dtype=case.dtype, device=self.device)
        return (A,)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    bench = SlogdetBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg.slogdet,
        gems_op=flag_gems.linalg_slogdet,
        # linalg.slogdet generated kernel only supports float32 on CUDA.
        dtypes=[torch.float32],
    )
    bench.run()
