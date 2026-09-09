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

# abs is a unary pointwise op, so the public UnaryPointwiseBenchmark family
# covers its semantics (input shapes from core_shapes.yaml). The candidate is
# passed explicitly via gems_op and the perf reference is the aten op itself
# (torch.ops.aten.abs / abs_); both are called with the same single-tensor
# signature that the family's build_inputs produces.


@pytest.mark.abs
def test_abs():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs",
        torch_op=torch.ops.aten.abs,
        gems_op=flag_gems.abs,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.abs_
def test_abs_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs_",
        torch_op=torch.ops.aten.abs_,
        gems_op=flag_gems.abs_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
