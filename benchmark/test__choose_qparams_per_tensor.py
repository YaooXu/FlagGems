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

import os
import sys

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import base`` cannot resolve this checkout's benchmark package through
# normal package discovery. Put the checkout root on sys.path so the
# ``benchmark`` package resolves to THIS checkout no matter how pytest is
# invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts, utils  # noqa: E402

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

# aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False)
# -> (float, int) computes a per-tensor min/max reduction and returns a Python
# (float, int) pair. There is no core_shapes.yaml entry for it, so the base
# class would fall back to consts.DEFAULT_SHAPES, which includes a 1-B-element
# 1-D tensor whose allocation cost would dominate the measurement. Use a
# modest, allocation-friendly shape set instead.
CQPT_SHAPES = [
    (65536,),
    (1_048_576,),
    (4096, 1024),
]


def _case_fn(shape, dtype):
    del dtype
    for reduce_range in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"reduce_range": reduce_range},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    # unpack_to_args_kwargs turns the params dict into call kwargs:
    # op(input, reduce_range=...).
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
