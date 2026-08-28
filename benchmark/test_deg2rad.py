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


def _deg2rad_out_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "out": shape},
        params={"out_aliases_input": True},
        builder_args=(shape,),
    )


def _deg2rad_out_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = base.generate_tensor_input(shape, dtype, device)
    return inp, {"out": inp}


@pytest.mark.deg2rad
def test_deg2rad():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad",
        torch_op=torch.deg2rad,
        gems_op=flag_gems.deg2rad,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.deg2rad_
def test_deg2rad_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad_",
        torch_op=lambda x: x.deg2rad_(),
        gems_op=flag_gems.deg2rad_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.deg2rad_out
def test_deg2rad_out():
    bench = base.GenericBenchmark(
        op_name="deg2rad_out",
        case_fn=_deg2rad_out_case_fn,
        build_inputs_fn=_deg2rad_out_build_inputs_fn,
        torch_op=torch.deg2rad,
        gems_op=flag_gems.deg2rad_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
