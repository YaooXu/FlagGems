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

# ``_fw_primal`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._fw_primal`` and ``-m _fw_primal`` both
# work.
setattr(
    pytest.mark,
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# Make sure the FlagGems checkout that physically contains this file is the one
# used for the sibling ``benchmark`` package. Under pytest
# ``--import-mode=importlib`` the process sys.path may hold an unrelated entry
# that shadows this checkout's ``benchmark`` package; insert the checkout root
# at the front and re-import the package from this file's own directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg

from . import base, consts, utils  # noqa: E402

# aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a) is a zero-copy
# forward-mode-AD view: it shares the input storage and allocates nothing, so
# the benchmark measures dispatch and view-construction overhead. The default
# shape set contains a 1-B-element 1-D tensor whose cost would be dominated by
# input allocation; use allocation-friendly shapes instead (the largest case
# below is 128 MiB fp32, so the two tensor inputs of the timing harness fit
# comfortably on busy GPUs).
FW_PRIMAL_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 512, 512),
]


def _case_fn(shape, dtype):
    del dtype
    # The op needs a ``level`` alongside the tensor, which the public unary
    # pointwise families do not supply, so the two-phase GenericBenchmark
    # forwards the constant level explicitly (documented level 0).
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": 0},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, level = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"level": level}


class FwPrimalBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = FW_PRIMAL_SHAPES


@pytest.mark._fw_primal
def test__fw_primal():
    bench = FwPrimalBenchmark(
        op_name="_fw_primal",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._fw_primal,
        gems_op=getattr(flag_gems, "_fw_primal", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
