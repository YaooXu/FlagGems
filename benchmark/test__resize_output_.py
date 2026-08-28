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

import flag_gems
import pytest
import torch

from . import base, consts

# Benchmark for _resize_output_
_RESIZE_OUTPUT_SHAPES = [
    (1024,),
    (2048,),
    (4096,),
    (1024, 1024),
    (2048, 512),
    (512, 2048),
]


class ResizeOutputBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return _RESIZE_OUTPUT_SHAPES


def _case_fn(shape, dtype):
    del dtype
    numel = math.prod(shape)
    # Target size - same number of elements but different shape when possible
    if numel == 1024:
        target_size = [32, 32]
    elif numel == 2048:
        target_size = [64, 32]
    elif numel == 4096:
        target_size = [64, 64]
    elif numel == 1024 * 1024:
        target_size = [1024, 1024]
    elif numel == 2048 * 512:
        target_size = [2048, 512]
    elif numel == 512 * 2048:
        target_size = [512, 2048]
    else:
        target_size = [numel]
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"target_size": target_size},
        builder_args=(shape, target_size),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, target_size = plan.builder_args
    # Create input tensor
    inp = torch.randn(*shape, device=device, dtype=dtype)
    return inp, target_size, {"device": device}


@pytest.mark.resize_output_
def test_resize_output_():
    # Note: PyTorch doesn't have _resize_output_ implemented for CUDA, so we use
    # a dummy torch_op that calls our gems implementation as the "baseline"
    def dummy_torch_op(inp, size, device):
        # Use the same in-place implementation as GEMS for baseline
        return flag_gems._resize_output_(inp, size, device)

    bench = ResizeOutputBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="resize_output_",
        torch_op=dummy_torch_op,
        gems_op=flag_gems._resize_output_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
