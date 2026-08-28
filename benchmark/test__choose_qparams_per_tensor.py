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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts, utils

# ``_choose_qparams_per_tensor`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._choose_qparams_per_tensor``
# and ``-m _choose_qparams_per_tensor`` both work.
setattr(
    pytest.mark,
    "_choose_qparams_per_tensor",
    MarkDecorator(
        Mark("_choose_qparams_per_tensor", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_choose_qparams_per_tensor computes a per-tensor min/max reduction and
# returns a Python (float, int) pair. There is no core_shapes.yaml entry for it,
# so the base class would fall back to consts.DEFAULT_SHAPES, which includes a
# 1-B-element 1-D tensor whose allocation cost would dominate the measurement.
# Use a modest, allocation-friendly shape set instead.
CQPT_SHAPES = [
    (65536,),
    (1_048_576,),
    (4096, 1024),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"reduce_range": False},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    # unpack_to_args_kwargs turns the params dict into call kwargs:
    # op(input, reduce_range=False).
    return inp, {"reduce_range": plan.params["reduce_range"]}


class ChooseQParamsPerTensorBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = CQPT_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor():
    bench = ChooseQParamsPerTensorBenchmark(
        op_name="_choose_qparams_per_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._choose_qparams_per_tensor,
        gems_op=getattr(flag_gems, "_choose_qparams_per_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
