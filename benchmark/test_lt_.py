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


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


def _case_fn_scalar(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"scalar": 0},
        builder_args=(shape, 0),
    )


@pytest.mark.lt_
def test_lt_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt_",
        torch_op=lambda a, b: torch.ops.aten.lt_.Tensor(a, b),
        gems_op=flag_gems.lt_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.lt_scalar_
def test_lt_scalar_():
    bench = base.GenericBenchmark(
        op_name="lt_scalar_",
        case_fn=_case_fn_scalar,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn_scalar),
        torch_op=lambda a, b: torch.ops.aten.lt_.Scalar(a, b),
        gems_op=flag_gems.lt_scalar_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
