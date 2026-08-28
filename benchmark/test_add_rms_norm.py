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


def _add_rms_norm_case_fn(shape, dtype):
    del dtype
    M, N = shape
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "weight": [N]},
        params={},
        builder_args=(shape,),
    )


def _add_rms_norm_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    M, N = shape
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    return inp1, inp2, (N,), weight


@pytest.mark.add_rms_norm
def test_add_rms_norm():
    # Use a custom wrapper for torch implementation
    def torch_add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
        x = x1 + x2
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    bench = base.GenericBenchmark2DOnly(
        case_fn=_add_rms_norm_case_fn,
        build_inputs_fn=_add_rms_norm_build_inputs_fn,
        op_name="add_rms_norm",
        torch_op=torch_add_rms_norm,
        gems_op=flag_gems.add_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
