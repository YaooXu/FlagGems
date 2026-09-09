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

import itertools

import pytest
import torch

import flag_gems

from . import base, consts

# aten::crow_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (nrows + 1,) int64 compressed row index tensor of a sparse CSR
# tensor as a fresh, contiguous, independent copy (the view_copy counterpart of
# aten::crow_indices, whose native body is
# `crow_indices(self).clone(contiguous)`). Its cost is proportional to the size
# of the returned tensor and independent of the stored values, so benchmark a
# spread of logical shapes (nrows, ncols) / (batch, nrows, ncols) /
# (batch_1, batch_2, nrows, ncols) and nnz values. The device-side allocation
# stays small relative to the logical size because only nnz entries are stored.
_CROW_COPY_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((16, 1024, 1024), 262144),
    ((8, 256, 256), 1048576),
    ((64, 1024, 1024), 131072),
    ((4, 8, 256, 256), 262144),
]


def _torch_crow_indices_copy(inp):
    # torch.ops.aten.crow_indices_copy is registered as
    # CompositeExplicitAutogradNonFunctional, whose dispatch-key set can exclude
    # the SparseCsr functionality key, so it may be unreachable on sparse CSR
    # tensors in some builds. Benchmark the operator's exact native body —
    # crow_indices(self).clone(contiguous) — which shares call semantics with
    # the candidate. Prefer the literal ATen op when it is reachable.
    try:
        return torch.ops.aten.crow_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.crow_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _make_structure(logical_shape, nnz, device):
    # Random (row, col) structure with a valid crow pointer array, generated
    # directly on the benchmark device. (row, col) pairs are drawn with
    # replacement; the crow array is built with a row-wise bincount so the
    # result is always a valid CSR structure.
    nrows, ncols = logical_shape[-2], logical_shape[-1]
    batch = logical_shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, device=device)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, device=device)
    order = torch.argsort(rows * ncols + cols, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    counts = torch.stack(
        [
            torch.bincount(rows[idx], minlength=nrows)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (nrows,))
    crow = torch.zeros(batch + (nrows + 1,), dtype=torch.long, device=device)
    crow[..., 1:] = torch.cumsum(counts, -1)
    return crow, cols


def _case_fn(shape, dtype):
    del dtype
    logical_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": logical_shape},
        params={"nnz": nnz},
        builder_args=(logical_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    logical_shape, nnz = plan.builder_args
    crow, cols = _make_structure(logical_shape, nnz, device)
    values_shape = logical_shape[:-2] + (nnz,)
    values = torch.randn(values_shape, dtype=dtype, device=device)
    inp = torch.sparse_csr_tensor(crow, cols, values, logical_shape)
    return inp, {}


class CrowIndicesCopyBenchmark(base.GenericBenchmark):
    # crow_indices_copy is a sparse metadata accessor; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated (logical_shape,
    # nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _CROW_COPY_SHAPES


@pytest.mark.crow_indices_copy
def test_crow_indices_copy():
    bench = CrowIndicesCopyBenchmark(
        op_name="crow_indices_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=_torch_crow_indices_copy,
        gems_op=getattr(flag_gems, "crow_indices_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
