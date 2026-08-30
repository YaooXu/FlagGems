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
# ``from . import base, consts`` cannot resolve this checkout's benchmark
# package through normal package discovery. Put the checkout root on sys.path so
# the ``benchmark`` package resolves to THIS checkout no matter how pytest is
# invoked (belt-and-suspenders: the correctness file already does this when it
# runs first, but this keeps the benchmark file self-contained).
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# sparse_mask gathers values from a dense ``self`` at a sparse mask's index
# positions and returns a sparse COO tensor. There is no public Benchmark family
# for sparse gather ops, so the benchmark uses the two-phase GenericBenchmark
# (case_fn + build_inputs_fn). The candidate is resolved at run time from the
# process-local override (flag_gems.testing.resolve_gems_op) via GenericBenchmark's
# _resolve_direct_gems_op; flag_gems.sparse_mask does not exist yet as an
# attribute, so the direct-callable default is fetched with getattr and may be
# None. The perf reference is the aten op itself (torch.ops.aten.sparse_mask)
# and both are called with the same (self, mask) signature.
_SPARSE_MASK_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


def _case_fn(shape, dtype):
    del dtype
    # yield generates one BenchmarkCasePlan per (shape, dtype) parametrization.
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "mask": shape},
        params={"nnz_ratio": 0.1},
        builder_args=(shape, 0.1),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, nnz_ratio = plan.builder_args
    self_t = torch.randn(shape, dtype=dtype, device=device)
    mask_dense = torch.rand(shape, device=device) > (1.0 - nnz_ratio)
    mask = mask_dense.to_sparse()
    return self_t, mask, {}


class SparseMaskBenchmark(base.GenericBenchmark):
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
