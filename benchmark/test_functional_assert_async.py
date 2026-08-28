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

# _functional_assert_async only operates on single-element tensors, so the
# shapes are all unit-sized. The dependency token semantics are exercised
# per dtype below.
FUNCTIONAL_ASSERT_ASYNC_SHAPES = [
    (1,),
    (1, 1),
    (1, 1, 1),
]


class FunctionalAssertAsyncBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = FUNCTIONAL_ASSERT_ASYNC_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"tensor": list(shape)},
                    params={"assert_msg": "Benchmark functional_assert_async"},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        # Non-zero single-element tensor so the device assertion passes.
        tensor = torch.ones(shape, dtype=case.dtype, device=self.device)
        dep_token = torch.empty(0, dtype=case.dtype, device=self.device)
        return tensor, plan.params["assert_msg"], dep_token


@pytest.mark.functional_assert_async
def test_functional_assert_async():
    bench = FunctionalAssertAsyncBenchmark(
        op_name="functional_assert_async",
        # Use flag_gems._functional_assert_async for both baseline and gems
        # since there is no native PyTorch CUDA implementation for this op.
        torch_op=flag_gems._functional_assert_async,
        gems_op=flag_gems._functional_assert_async,
        dtypes=[torch.int32, torch.float32, torch.float16],
    )
    bench.run()
