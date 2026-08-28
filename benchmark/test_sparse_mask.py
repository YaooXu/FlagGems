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
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts  # noqa: E402

# aten::sparse_mask(self, mask) gathers values of ``self`` at the nonzero
# positions of ``mask``, so its cost scales with the mask's nnz rather than the
# dense element count. There are no meaningful dense entries for it in
# core_shapes.yaml, so benchmark dedicated (dense shape, nnz_ratio) pairs.
_SPARSE_MASK_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "mask": shape},
        params={"nnz_ratio": 0.1},
        builder_args=(shape, 0.1),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, nnz_ratio = plan.builder_args
    self_t = torch.randn(shape, dtype=dtype, device=device)
    mask = (torch.rand(shape, device=device) > nnz_ratio).to_sparse()
    return self_t, mask, {}


class SparseMaskBenchmark(base.GenericBenchmark):
    # sparse_mask is a sparse gather op; override set_shapes so the core shape
    # file lookup is bypassed and the dedicated (shape, nnz_ratio) pairs above
    # are used directly.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _SPARSE_MASK_SHAPES


@pytest.mark.sparse_mask
def test_sparse_mask():
    bench = SparseMaskBenchmark(
        op_name="sparse_mask",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_mask,
        gems_op=getattr(flag_gems, "sparse_mask", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
