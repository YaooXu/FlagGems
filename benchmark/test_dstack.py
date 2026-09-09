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

from . import base, consts, utils

# aten::dstack(Tensor[] tensors) -> Tensor views every input as 3-D (atleast_3d)
# and concatenates along the new depth axis (dim 2); the output is ~3x the input
# size for a 3-element TensorList. No public Benchmark family models a
# TensorList depth-concatenation, so the benchmark uses a two-phase
# GenericBenchmark (case_fn + build_inputs_fn). The benchmark trips every input
# three times (3 tensors of the same shape) to keep the depth-axis copy
# dominant; the cap on the total element count keeps allocations reasonable.
# gems_op is resolved through getattr because flag_gems.dstack is not yet
# registered as a direct callable; KernelGen's override_gems_op("dstack", ...)
# still wins at run time via flag_gems.testing.resolve_gems_op.


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"inputs": [shape, shape, shape]},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = [utils.generate_tensor_input(shape, dtype, device) for _ in range(3)]
    return inp, {}


class DstackBenchmark(base.GenericBenchmark):
    # The depth-axis copy is the dominant cost, so capping the total element
    # count avoids allocating multi-GB inputs (the generic DEFAULT_SHAPES
    # include 1G-element tensors) for no signal: a 3-tensor dstack writes
    # 3x the input elements, so the cap is 3 * 2**26 elements.
    MAX_ELEMENTS = 3 * 2**26

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if _numel(shape) * 3 <= self.MAX_ELEMENTS
        ]

    def set_more_shapes(self):
        # Depth-axis performance-relevant shapes: 1-D (2**20,), 2-D rows of
        # width 2**i, and 3-D (64, 2**i, 64) volumes whose depth axis is the
        # concatenation dimension.
        self.shapes = self.shapes + [
            (2**20,),
            (1024, 2**0),
            (1024, 2**8),
            (1024, 2**12),
            (64, 2**0, 64),
            (64, 2**4, 64),
            (64, 2**8, 64),
        ]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


@pytest.mark.dstack
@pytest.mark.dstack_benchmark
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
