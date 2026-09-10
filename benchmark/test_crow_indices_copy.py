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

import math

import pytest
import torch

import flag_gems

from . import base, consts

# (layout, size, nnz, blocks). crow_indices_copy materializes the
# batch_dims + (n_compressed + 1,) compressed-row pointer array of a sparse CSR
# or BSR tensor as a fresh contiguous int64 copy. It is a metadata accessor
# whose cost is proportional to the number of compressed rows (the size of the
# returned array) and independent of the stored values, so the benchmark sweeps
# a spread of row extents, batch dims and block sizes. Only nnz entries are
# stored on the device, keeping the values allocation small relative to the
# logical size.
_CROW = [
    ("csr", (1024, 1024), 65536, None),
    ("csr", (4096, 4096), 1048576, None),
    ("csr", (65536, 1024), 1048576, None),
    ("csr", (262144, 64), 1048576, None),
    ("csr_batch", (8, 4096, 4096), 131072, None),
    ("bsr", (4096, 4096), 262144, (8, 8)),
    ("bsr", (8192, 8192), 65536, (16, 16)),
    ("bsr_batch", (4, 4096, 4096), 65536, (8, 8)),
]


def _random_values(shape, dtype, gen):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    return torch.randint(-5, 6, shape, dtype=dtype, generator=gen)


def _random_crow(n_compressed, nnz, gen):
    # Non-decreasing compressed-row array of length n_compressed + 1 with
    # crow[0] == 0 and crow[-1] == nnz.
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            inner,
            torch.tensor([nnz], dtype=torch.long),
        ]
    )


def _make_input(layout, size, nnz, blocks, dtype, device):
    gen = torch.Generator("cpu").manual_seed(0)
    if layout == "csr":
        n_rows, n_cols = size
        crow = _random_crow(n_rows, nnz, gen)
        col = torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen)
        values = _random_values((nnz,), dtype, gen)
        return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)
    if layout == "csr_batch":
        batch, n_rows, n_cols = size
        crows, cols, values = [], [], []
        for _ in range(batch):
            crows.append(_random_crow(n_rows, nnz, gen))
            cols.append(
                torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen)
            )
            values.append(_random_values((nnz,), dtype, gen))
        return torch.sparse_csr_tensor(
            torch.stack(crows),
            torch.stack(cols),
            torch.stack(values),
            size=size,
            device=device,
        )
    if layout == "bsr":
        n_rows, n_cols = size
        br, bc = blocks
        n_row_blocks = math.ceil(n_rows / br)
        n_col_blocks = math.ceil(n_cols / bc)
        crow = _random_crow(n_row_blocks, nnz, gen)
        col = torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
        values = _random_values((nnz, br, bc), dtype, gen)
        return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)
    if layout == "bsr_batch":
        batch, n_rows, n_cols = size
        br, bc = blocks
        n_row_blocks = math.ceil(n_rows / br)
        n_col_blocks = math.ceil(n_cols / bc)
        crows, cols, values = [], [], []
        for _ in range(batch):
            crows.append(_random_crow(n_row_blocks, nnz, gen))
            cols.append(
                torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
            )
            values.append(_random_values((nnz, br, bc), dtype, gen))
        return torch.sparse_bsr_tensor(
            torch.stack(crows),
            torch.stack(cols),
            torch.stack(values),
            size=size,
            device=device,
        )
    raise ValueError(f"unknown layout {layout}")


def _torch_crow_indices_copy(inp):
    # torch_op is the perf comparison reference and shares call semantics with
    # the candidate: the real ATen operator, probed invocable on sparse CSR/BSR
    # tensors, so no composed simulation is used.
    return torch.ops.aten.crow_indices_copy(inp)


def _gems_crow_indices_copy(inp):
    # Resolved through the direct-callable route (override-aware) rather than
    # going through the dispatcher. getattr keeps this importable while
    # flag_gems has no public crow_indices_copy attribute yet.
    op = flag_gems.testing.resolve_gems_op(
        "crow_indices_copy", getattr(flag_gems, "crow_indices_copy", None)
    )
    return op(inp)


def _case_fn(shape, dtype):
    del dtype
    layout, size, nnz, blocks = shape
    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"nnz": nnz, "layout": layout},
        builder_args=(layout, size, nnz, blocks),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, size, nnz, blocks = plan.builder_args
    inp = _make_input(layout, size, nnz, blocks, dtype, device)
    return inp, {}


class CrowIndicesCopyBenchmark(base.GenericBenchmark):
    # crow_indices_copy is a sparse metadata accessor; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated
    # (layout, size, nnz, blocks) cases instead.
    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        self.shapes = _CROW


@pytest.mark.crow_indices_copy
def test_crow_indices_copy():
    bench = CrowIndicesCopyBenchmark(
        op_name="crow_indices_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=_torch_crow_indices_copy,
        gems_op=_gems_crow_indices_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
