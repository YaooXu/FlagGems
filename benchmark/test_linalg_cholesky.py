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

fp64_is_supported = flag_gems.runtime.device.support_fp64

# Cholesky decomposition benchmark shapes
# Square matrices from 2x2 to 256x256 covering small to medium-large use cases
CHOLESKY_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


class CholeskyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SHAPES

    def get_input_iter(self, cur_dtype):
        for case in self.get_case_iter(cur_dtype):
            yield self.build_inputs(case)

    def supports_cases(self) -> bool:
        return type(self).get_input_iter is CholeskyBenchmark.get_input_iter

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        n = shape[-1]
        # Create the same positive-definite input as the legacy benchmark.
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        positive_definite = (
            matrix @ matrix.transpose(-2, -1)
            + torch.eye(n, dtype=case.dtype, device=self.device) * 0.1
        )
        return (positive_definite,)


@pytest.mark.linalg_cholesky
def test_linalg_cholesky():
    bench = CholeskyBenchmark(
        op_name="linalg_cholesky",
        torch_op=torch.ops.aten.linalg_cholesky,
        gems_op=flag_gems.linalg_cholesky,
        # Cholesky only supports float32/float64; fp16/bf16 not supported by
        # PyTorch. fp64 is gated on device support (Moore Threads has no fp64).
        dtypes=[torch.float32] + ([torch.float64] if fp64_is_supported else []),
    )
    bench.run()
