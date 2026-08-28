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

import math

import pytest
import torch

import flag_gems

from . import base, consts


class ViewCopyBenchmark(base.UnaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            size = (-1,)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape, "output": (math.prod(shape),)},
                    params={"size": size},
                    builder_args=(shape, size),
                ),
            )

    def build_inputs(self, case):
        shape, size = case.builder_args[0].builder_args
        inp = base.generate_tensor_input(shape, case.dtype, self.device)
        return inp, size


@pytest.mark.view_copy
def test_view_copy():
    bench = ViewCopyBenchmark(
        op_name="view_copy",
        torch_op=torch.view_copy,
        gems_op=flag_gems.view_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
