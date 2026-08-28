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

# aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *,
#     ScalarType? dtype, ...) -> Tensor constructs a sparse COO tensor of the
# given ``size`` from the raw (sparse_dim, nnz) int64 index tensor and the
# (nnz,) + dense values tensor. The measured work is the layout construction
# from the three components, so the benchmark feeds the components directly
# (not a pre-built sparse tensor) and both the reference and the candidate
# receive the exact same call. The size-only and size-inferred overloads are
# trivial (empty / max+1) and are not benchmarked.
#
# Each benchmark case is (tensor_shape, nnz): coordinates are drawn with
# replacement per sparse dim (kept in random order, so the constructed tensor
# is uncoalesced), so the storage cost grows with nnz while the logical matrix
# spans the full (rows, cols) extent.
_BENCH_SHAPES = [
    ((4096, 4096), 10000),
    ((4096, 4096), 100000),
    ((1024, 8192), 100000),
    ((8192, 1024), 100000),
    ((4096, 4096), 1000000),
    ((2048, 2048, 64), 100000),
]


def _make_coo_inputs(shape, nnz, dtype, device, seed=0):
    # Deterministic CPU-side generation of a valid (indices, values) pair for
    # the explicit-size overload: int64 indices of shape (sparse_dim, nnz) with
    # coordinates drawn with replacement (kept in random order, so the
    # constructed tensor is uncoalesced) and values of shape (nnz, dense...).
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_dim = min(2, len(shape))
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    entries = []
    for d in range(sparse_dim):
        entries.append(torch.randint(0, sparse_shape[d], (nnz,), generator=gen))
    indices = torch.stack(entries, dim=0)
    values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    return indices.to(device), values.to(device)


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz},
        builder_args=(tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz = plan.builder_args
    indices, values = _make_coo_inputs(tensor_shape, nnz, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner, so
    # torch_op and gems_op both receive (indices, values, size, dtype=...,
    # device=...). The dtype keyword is required: without it the aten op forces
    # float32 and would break the bfloat16/float16 cases.
    return indices, values, list(tensor_shape), {"dtype": dtype, "device": device}


class SparseCooTensorBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark feeding the raw COO components to the
    ``sparse_coo_tensor`` factory call."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _BENCH_SHAPES


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor():
    bench = SparseCooTensorBenchmark(
        op_name="sparse_coo_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_coo_tensor,
        gems_op=getattr(flag_gems, "sparse_coo_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
