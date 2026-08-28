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

from . import base, consts

# ``_efficientzerotensor`` starts with an underscore, and ``pytest.mark`` refuses
# to generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._efficientzerotensor`` and
# ``-m _efficientzerotensor`` both work.
for _name in ("_efficientzerotensor", "_efficientzerotensor_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_efficientzerotensor is a zero-copy factory: on CUDA it returns an
# all-zero tensor backed by a single shared zero byte (nbytes == 0), so the
# default-variant benchmark measures dispatch + storage-construction overhead
# rather than memory bandwidth. The default shape set contains a 1-B-element
# 1-D tensor whose cost would be dominated by input allocation; use
# allocation-friendly shapes that still exercise a realistic range of ranks.
# The .out variant writes a real zero-fill, so these shapes keep the fill work
# bounded as well.
EFFICIENTZEROTENSOR_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (20, 320, 15),
    (16, 128, 64, 1280),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    return shape, {"dtype": dtype, "device": device}


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    out = torch.empty(shape, dtype=dtype, device=device)
    return shape, {"out": out}


class EfficientZeroTensorBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes.

    The default shape set contains a 1-B-element 1-D tensor whose cost would be
    dominated by input allocation, so the case list is restricted to the shapes
    above.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = EFFICIENTZEROTENSOR_SHAPES


@pytest.mark._efficientzerotensor
def test__efficientzerotensor():
    bench = EfficientZeroTensorBenchmark(
        op_name="_efficientzerotensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._efficientzerotensor,
        gems_op=getattr(flag_gems, "_efficientzerotensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark._efficientzerotensor_out
def test__efficientzerotensor_out():
    bench = EfficientZeroTensorBenchmark(
        op_name="_efficientzerotensor_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._efficientzerotensor.out,
        gems_op=getattr(flag_gems, "_efficientzerotensor_out", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
