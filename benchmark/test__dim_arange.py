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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# ``_dim_arange`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._dim_arange`` and ``-m _dim_arange`` both
# work.
setattr(
    pytest.mark,
    "_dim_arange",
    MarkDecorator(Mark("_dim_arange", (), {}, _ispytest=True), _ispytest=True),
)

# The KernelGen verification harness stages this file in a temporary copy of the
# FlagGems tree and runs pytest in-process with ``--import-mode=importlib`` from
# that temp root, which is not placed on ``sys.path``. Bootstrap the checkout
# root from ``__file__`` so the relative imports below resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import base, consts, utils  # noqa: E402

# aten::_dim_arange(like, dim) builds a fresh 1-D int64 tensor of length
# like.size(dim). The measured work is proportional to the extent of the
# selected dim (plus the input allocation), so each benchmark shape is paired
# with its largest extent. The 1-D shapes dominate the arange materialization
# itself; the multi-dim shapes cover the common "arange along one axis" usage.
_DIM_ARANGE_SHAPES = [
    (2**20,),
    (2**24,),
    (2**16, 16),
    (4096, 4096),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]


def _case_fn(shape, dtype):
    del dtype
    dim = max(range(len(shape)), key=lambda d: shape[d])
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dim": dim},
        builder_args=(shape, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"dim": dim}


class DimArangeBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark with shapes tuned for _dim_arange."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _DIM_ARANGE_SHAPES


@pytest.mark._dim_arange
def test__dim_arange():
    bench = DimArangeBenchmark(
        op_name="_dim_arange",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._dim_arange,
        gems_op=getattr(flag_gems, "_dim_arange", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
