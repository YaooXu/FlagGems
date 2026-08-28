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


def addcdiv__input_fn(shape, dtype, device):
    # For in-place addcdiv_, we need to yield the arguments for tensor.addcdiv_() method call
    # The input function format: (inp1, inp2, inp3) + kwargs
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


def addcdiv__case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "tensor1": shape, "tensor2": shape},
        params={"value": 0.5},
        builder_args=(shape, 0),
    )


@pytest.mark.addcdiv_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_addcdiv_():
    bench = base.GenericBenchmark(
        op_name="addcdiv_",
        torch_op=torch.Tensor.addcdiv_,
        gems_op=flag_gems.addcdiv_,
        case_fn=addcdiv__case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(addcdiv__input_fn),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
