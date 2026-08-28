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

# Benchmark shapes for sym_stride - covering various tensor dimensionalities
SYM_STRIDE_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


class SymStrideBenchmark(base.Benchmark):
    """Custom benchmark for sym_stride - returns tensor metadata (stride), not a computed tensor."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SYM_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype):
        for case in self.get_case_iter(cur_dtype):
            yield self.build_inputs(case)

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return (torch.randn(shape, dtype=case.dtype, device=self.device),)


@pytest.mark.sym_stride
def test_sym_stride():
    bench = SymStrideBenchmark(
        op_name="sym_stride",
        torch_op=torch.ops.aten.sym_stride,
        gems_op=flag_gems.sym_stride,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
