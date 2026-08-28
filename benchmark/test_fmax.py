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


@pytest.mark.fmax
def test_fmax():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmax",
        torch_op=torch.fmax,
        gems_op=flag_gems.fmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _fmax_out_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _fmax_out_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    return inp1, inp2, {"out": out}


@pytest.mark.fmax_out
def test_fmax_out():
    bench = base.GenericBenchmark(
        op_name="fmax_out",
        torch_op=torch.fmax,
        case_fn=_fmax_out_case_fn,
        build_inputs_fn=_fmax_out_build_inputs_fn,
        gems_op=flag_gems.fmax_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
