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

from . import base


def _input_fn(shape, dtype, device):
    yield {"s": 0.01, "dtype": dtype, "device": device},


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={},
        params={"s": 0.01},
        builder_args=(shape, 0),
    )


@pytest.mark.scalar_tensor
def test_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="scalar_tensor",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn),
        torch_op=torch.scalar_tensor,
        gems_op=flag_gems.scalar_tensor,
    )
    bench.run()
