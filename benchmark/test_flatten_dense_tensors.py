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

import sys as _sys
from pathlib import Path as _Path

import pytest
import torch

import flag_gems

# The KernelGen integration harness verifies this file inside a temporary copy
# of the FlagGems tree. That process is launched with sys.path[0] pointing at
# the harness script, not the tree root, so the parent `benchmark` package
# would not resolve. Insert the tree root so the relative import below always
# works.
_TREE_ROOT = str(_Path(__file__).resolve().parents[1])
if _TREE_ROOT not in _sys.path:
    _sys.path.insert(0, _TREE_ROOT)

from . import base, consts, utils  # noqa: E402

# aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor flattens every input
# to a contiguous 1-D tensor and concatenates them into one 1-D result. It is a
# bandwidth-bound data-movement op (copy + cat), so each case is a list of
# tensor shapes whose total element count is performance-relevant
# (1M - 12.5M elements).
FLATTEN_DENSE_TENSORS_SHAPES = [
    [(1024, 1024)],
    [(4096, 4096)],
    [(1024, 1024), (1024, 1024), (1024, 1024)],
    [(64, 512, 512)],
    [(2048, 2048), (2048, 2048)],
    [(16384, 256), (16384, 256), (16384, 256)],
]


def _case_fn(shape, dtype):
    del dtype
    numel = 0
    for s in shape:
        numel += torch.Size(s).numel()
    yield base.BenchmarkCasePlan(
        shape={"tensors": shape},
        params={"numel": numel},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (tensor_shapes,) = plan.builder_args
    tensors = [
        utils.generate_tensor_input(shape, dtype, device) for shape in tensor_shapes
    ]
    return tensors, {}


class FlattenDenseTensorsBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to list-of-shapes cases."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = FLATTEN_DENSE_TENSORS_SHAPES

    def set_more_shapes(self):
        # Every case consumes a list of tensors, so the framework's bare-tuple
        # comprehensive shapes do not apply.
        return []


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors():
    bench = FlattenDenseTensorsBenchmark(
        op_name="flatten_dense_tensors",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.flatten_dense_tensors,
        gems_op=getattr(flag_gems, "flatten_dense_tensors", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
