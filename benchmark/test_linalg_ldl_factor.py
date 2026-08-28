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


# LDL Factorization benchmark
class LdlFactorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # LDL factorization shapes (square matrices only)
        self.shapes = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
        ]

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"matrix_intent": "symmetric_positive_definite"},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        n = shape[0]
        A = torch.randn(shape, dtype=case.dtype, device=self.device)
        A = (
            A @ A.transpose(-2, -1)
            + torch.eye(n, dtype=case.dtype, device=self.device) * n
        )
        return (A,)


@pytest.mark.linalg_ldl_factor
def test_linalg_ldl_factor():
    bench = LdlFactorBenchmark(
        op_name="linalg_ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        gems_op=flag_gems.ldl_factor,
        # torch.linalg.ldl_factor on CUDA supports float32/float64 for this path.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
