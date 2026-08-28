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


@pytest.mark.softplus
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _softplus_backward_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"grad_output": shape, "input": shape},
        params={"beta": 1.0, "threshold": 20.0},
        builder_args=(shape,),
    )


def _softplus_backward_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    grad_output = base.generate_tensor_input(shape, dtype, device)
    inp = base.generate_tensor_input(shape, dtype, device)
    return grad_output, inp, plan.params["beta"], plan.params["threshold"]


@pytest.mark.softplus_backward
def test_softplus_backward():
    bench = base.GenericBenchmark(
        op_name="softplus_backward",
        case_fn=_softplus_backward_case_fn,
        build_inputs_fn=_softplus_backward_build_inputs_fn,
        torch_op=torch.ops.aten.softplus_backward,
        gems_op=flag_gems.softplus_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
