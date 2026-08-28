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


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"reduction": 1},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = (
        torch.randint(0, 2, shape, device=device).to(dtype) * 2
    ) - 1
    grad_output = torch.ones(shape, dtype=dtype, device=device)
    return grad_output, inp, target, 1


@pytest.mark.soft_margin_loss_backward
def test_soft_margin_loss_backward():
    bench = base.GenericBenchmark(
        op_name="soft_margin_loss_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.soft_margin_loss_backward,
        gems_op=flag_gems.soft_margin_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
