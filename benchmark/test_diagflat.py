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

import flag_gems

# The KernelGen harness runs pytest in-process with ``--import-mode=importlib``,
# which does not prepend the checkout root to sys.path, so the ``benchmark``
# package may resolve to the harness's own package or not resolve at all.
# Re-point it at this file's directory before importing the benchmark helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts, utils  # noqa: E402

# aten::diagflat flattens the input (logical row-major) and writes it onto the
# diagonal of a square matrix with side length numel(input) + |offset|, so the
# output is quadratic in the input element count. The shape set is therefore
# bounded (~4096 input elements -> ~16M output elements) to keep the baseline
# and candidate allocations reasonable.
DIAGFLAT_SHAPES = [
    (64,),
    (256,),
    (1024,),
    (2048,),
    (4096,),
    (64, 64),
    (16, 16, 16),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"offset": 0},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, offset = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"offset": offset}


class DiagFlatBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to bounded input shapes.

    The default shape collection contains huge tensors whose quadratic diagflat
    output would exhaust device memory, so the case list is restricted to the
    small shapes above.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGFLAT_SHAPES
        self.shape_desc = "input numel (output side = numel + |offset|)"

    def set_more_shapes(self):
        return []


@pytest.mark.diagflat
def test_diagflat():
    bench = DiagFlatBenchmark(
        op_name="diagflat",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.diagflat,
        # KernelGen injects the candidate via override_gems_op(); the default
        # module callable may not exist until the op is merged into FlagGems.
        gems_op=getattr(flag_gems, "diagflat", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
