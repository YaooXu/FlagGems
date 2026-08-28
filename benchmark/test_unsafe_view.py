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

# Square 2D shapes covering common sizes for view benchmark
UNSAFE_VIEW_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


class UnsafeViewBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = UNSAFE_VIEW_SHAPES

    def get_input_iter(self, cur_dtype):
        for case in self.get_case_iter(cur_dtype):
            yield self.build_inputs(case)

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            new_shape = (shape[0] * shape[1],)
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape, "output": new_shape},
                    params={"size": new_shape},
                    builder_args=(shape, new_shape),
                ),
            )

    def build_inputs(self, case):
        shape, new_shape = case.builder_args[0].builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, new_shape


@pytest.mark.unsafe_view
def test_unsafe_view():
    bench = UnsafeViewBenchmark(
        op_name="unsafe_view",
        torch_op=torch.ops.aten._unsafe_view,
        gems_op=flag_gems._unsafe_view,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
