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

# Same package-resolution bootstrap as the correctness suite: the benchmark
# package that ships with this file must win over any other top-level
# ``benchmark`` package already importable on sys.path.
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_BENCH_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
_IMPORTED_BENCHMARK = sys.modules.get("benchmark")
if _IMPORTED_BENCHMARK is not None and os.path.abspath(
    getattr(_IMPORTED_BENCHMARK, "__file__", "")
) != os.path.join(_BENCH_DIR, "__init__.py"):
    del sys.modules["benchmark"]

from . import base, consts, utils  # noqa: E402

# ``_remove_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._remove_batch_dim`` and ``-m
# _remove_batch_dim`` both work.
setattr(
    pytest.mark,
    "_remove_batch_dim",
    MarkDecorator(Mark("_remove_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_remove_batch_dim inserts a broadcast batch dimension of size
# ``batch_size`` at position ``out_dim`` of the input shape (exactly
# ``self.expand(sizes)`` with ``batch_size`` inserted at ``out_dim``). It is a
# zero-copy view op, so the benchmark measures dispatch and view-construction
# overhead rather than memory traffic; the default shape set contains a
# 1-B-element 1-D tensor whose cost would be dominated by input allocation, so
# the allocation-friendly shapes below are used instead. Every case inserts the
# batch at out_dim=1 with batch_size matching the leading dim (the common vmap
# unwrap pattern).
REMOVE_BATCH_DIM_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 256),
    (20, 320, 15),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": 0, "batch_size": shape[0], "out_dim": 1},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {
        "level": plan.params["level"],
        "batch_size": plan.params["batch_size"],
        "out_dim": plan.params["out_dim"],
    }


class RemoveBatchDimBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = REMOVE_BATCH_DIM_SHAPES


@pytest.mark._remove_batch_dim
def test__remove_batch_dim():
    bench = RemoveBatchDimBenchmark(
        op_name="_remove_batch_dim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._remove_batch_dim,
        gems_op=getattr(flag_gems, "_remove_batch_dim", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
