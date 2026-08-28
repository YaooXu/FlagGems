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


def kthvalue_case_fn(shape, dtype):
    del dtype
    k = 2 if shape[-1] > 2 else shape[-1]
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"k": k, "dim": -1},
        builder_args=(shape,),
    )


def materialize_kthvalue_case(plan, dtype, device):
    x = torch.randn(plan.builder_args[0], device=device, dtype=dtype)
    return x, plan.params["k"], {"dim": plan.params["dim"]}


class KthvalueBenchmark(base.GenericBenchmarkExcluse1D):
    def set_shapes(self, shape_file_path=None):
        # 2D shapes for kthvalue along last dimension, exercising different dim sizes and batch sizes
        self.shapes = [
            (1024, 256),
            (4096, 64),
            (16384, 128),
            (512, 512),
            (2048, 512),
        ]


@pytest.mark.kthvalue
def test_kthvalue():
    bench = KthvalueBenchmark(
        op_name="kthvalue",
        torch_op=torch.kthvalue,
        gems_op=flag_gems.kthvalue,
        # Benchmark uses float32 only because topk gemm kernel operates in float32;
        # the kthvalue op auto-converts non-fp32 inputs internally.
        dtypes=[torch.float32],
        case_fn=kthvalue_case_fn,
        build_inputs_fn=materialize_kthvalue_case,
    )
    bench.run()
