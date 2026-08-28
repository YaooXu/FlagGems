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


def _special_multigammaln_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"p": 5},
        builder_args=(shape,),
    )


def _special_multigammaln_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    p = plan.params["p"]
    inp = base.generate_tensor_input(shape, dtype, device)
    return inp, p


@pytest.mark.special_multigammaln
def test_special_multigammaln():
    bench = base.GenericBenchmark(
        op_name="special_multigammaln",
        case_fn=_special_multigammaln_case_fn,
        build_inputs_fn=_special_multigammaln_build_inputs_fn,
        torch_op=torch.special.multigammaln,
        gems_op=flag_gems.special_multigammaln,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _mvlgamma_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"p": 5},
        builder_args=(shape,),
    )


def _mvlgamma_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    p = plan.params["p"]
    inp = utils.generate_tensor_input(shape, dtype, device)
    inp = inp.abs() + (p - 1) / 2 + 1.0
    return inp, p


@pytest.mark.mvlgamma_
def test_mvlgamma_():
    bench = base.GenericBenchmark(
        op_name="mvlgamma_",
        case_fn=_mvlgamma_case_fn,
        build_inputs_fn=_mvlgamma_build_inputs_fn,
        torch_op=torch.Tensor.mvlgamma_,
        gems_op=flag_gems.mvlgamma_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
