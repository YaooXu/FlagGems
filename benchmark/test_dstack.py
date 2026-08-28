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

from . import base, consts, utils


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"inputs": [shape, shape, shape]},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = [
        utils.generate_tensor_input(shape, dtype, device),
        utils.generate_tensor_input(shape, dtype, device),
        utils.generate_tensor_input(shape, dtype, device),
    ]
    return inp, {}


class DstackBenchmark(base.GenericBenchmark):
    # dstack materializes 3 input tensors + 1 output per case (4x one tensor's
    # memory), so cap the shape size to avoid OOM on the huge core shapes.
    MAX_ELEMENTS = 2**26

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]

    def set_more_shapes(self):
        # dstack reshapes 1-D (N,) -> (1, N, 1) and 2-D (M, N) -> (M, N, 1)
        # before concatenating along the new depth axis; cover these defining
        # paths alongside the surviving multi-dimensional core shapes.
        more_shapes_1d = [(2**20,)]
        more_shapes_2d = [(1024, 2**i) for i in (0, 8, 12)]
        more_shapes_3d = [(64, 2**i, 64) for i in (0, 4, 8)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


@pytest.mark.dstack
def test_dstack():
    bench = DstackBenchmark(
        op_name="dstack",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.dstack,
        gems_op=getattr(flag_gems, "dstack", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
