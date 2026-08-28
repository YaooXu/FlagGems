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


class AddmvBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []


def _case_fn(shape, dtype):
    m, n = shape
    yield base.BenchmarkCasePlan(
        shape={"mat": [m, n], "vec": [n], "bias": [m]},
        params={},
        builder_args=(m, n),
    )


def _build_inputs_fn(plan, dtype, device):
    m, n = plan.builder_args
    mat = torch.randn([m, n], dtype=dtype, device=device)
    vec = torch.randn([n], dtype=dtype, device=device)
    bias = torch.randn([m], dtype=dtype, device=device)
    # Tensor.addmv_(mat, vec)
    return bias, mat, vec


@pytest.mark.addmv_
def test_addmv_():
    bench = AddmvBenchmark(
        op_name="addmv_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.Tensor.addmv_,
        gems_op=flag_gems.addmv_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
