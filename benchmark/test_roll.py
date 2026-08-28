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


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)

    if len(shape) > 1:
        yield inp, {"shifts": (1, 2), "dims": (0, 1)}
    else:
        yield inp, {"shifts": 1, "dims": 0}


def _case_fn(shape, dtype):
    del dtype
    params = (
        {"shifts": (1, 2), "dims": (0, 1)}
        if len(shape) > 1
        else {"shifts": 1, "dims": 0}
    )
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params=params,
        builder_args=(shape, 0),
    )


@pytest.mark.roll
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_roll():
    bench = base.GenericBenchmark(
        op_name="roll",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn),
        torch_op=torch.roll,
        gems_op=flag_gems.roll,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
