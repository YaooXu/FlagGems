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


class ResizeBenchmark(base.UnaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            size = [math.prod(shape)]
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape, "output": size},
                    params={"size": size},
                    builder_args=(shape, tuple(size)),
                ),
            )

    def build_inputs(self, case):
        shape, size = case.builder_args[0].builder_args
        inp = base.generate_tensor_input(shape, case.dtype, self.device)
        return inp, list(size)


@pytest.mark.resize
def test_resize():
    bench = ResizeBenchmark(
        op_name="resize",
        torch_op=torch.ops.aten.resize,
        gems_op=flag_gems.resize,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.resize_
def test_resize_():
    bench = ResizeBenchmark(
        op_name="resize_",
        torch_op=torch.ops.aten.resize_,
        gems_op=flag_gems.resize_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
