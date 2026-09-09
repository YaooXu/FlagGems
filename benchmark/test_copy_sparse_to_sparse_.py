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

import flag_gems

from . import base, consts

# (sparse shape, sparse_dim, nnz). The copy transfers nnz entries, so timing
# cases sweep nnz from a small working set up to a few million stored entries,
# across 2-D, batched hybrid, all-sparse 3-D, and 4-D layouts. Every src keeps
# nnz >= numel so duplicate indices guarantee an uncoalesced source, which is
# the interesting storage for a verbatim structure copy.
_COPY_SPARSE_SHAPES = [
    ((1024, 1024), 2, 65536),
    ((1024, 1024), 2, 1048576),
    ((4096, 4096), 2, 1048576),
    ((64, 512, 512), 3, 524288),
    ((8, 256, 256), 3, 262144),
    ((16, 1024, 1024), 2, 8192),
    ((4, 8, 256, 256), 4, 262144),
]


def _make_sparse_input(shape, sparse_dim, nnz, dtype, device):
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in shape[:sparse_dim]
        ]
    )
    values = torch.randn((nnz,) + tuple(shape[sparse_dim:]), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(shape, dtype):
    del dtype
    shape, sparse_dim, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"nnz": nnz},
        builder_args=(shape, sparse_dim, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim, nnz = plan.builder_args
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype, device)
    dst = torch.zeros_like(src)
    return dst, src, {"non_blocking": False}


class CopySparseToSparseBenchmark(base.GenericBenchmark):
    # copy_sparse_to_sparse_ is a sparse op; there are no meaningful dense
    # shapes in core_shapes.yaml, so benchmark dedicated (shape, nnz) pairs
    # instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COPY_SPARSE_SHAPES


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_():
    bench = CopySparseToSparseBenchmark(
        op_name="copy_sparse_to_sparse_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.copy_sparse_to_sparse_,
        gems_op=getattr(flag_gems, "copy_sparse_to_sparse_", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
