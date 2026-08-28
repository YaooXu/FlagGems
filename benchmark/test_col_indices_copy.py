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

# KernelGen's verification harness stages the test files in a temporary
# copy of the FlagGems tree and runs pytest with --import-mode=importlib
# from a working directory that is not on sys.path, so the parent of this
# package may not be importable yet.  Make it importable before using the
# relative import below.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from . import base, consts  # noqa: E402

# (layout, size, nnz, blocks). col_indices_copy materializes the column index
# array of a sparse row-compressed tensor (CSR or BSR) as a fresh contiguous
# int64 copy -- a metadata accessor whose cost is proportional to nnz (times
# the batch size) and independent of the stored values, so benchmark a spread
# of nnz values, block sizes and layouts. The device-side allocation stays
# small relative to the logical size because only nnz entries (plus the tiny
# crow array) are stored.
_COLS = [
    ("csr", (1024, 1024), 65536, None),
    ("csr", (4096, 4096), 1048576, None),
    ("csr", (4096, 65536), 1048576, None),
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
    # Non-decreasing compressed row index array of length n_compressed + 1 with
    # crow[0] == 0 and crow[-1] == nnz.
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
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
        n_row_blocks = n_rows // br
        n_col_blocks = n_cols // bc
        crow = _random_crow(n_row_blocks, nnz, gen)
        col = torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
        values = _random_values((nnz, br, bc), dtype, gen)
        return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)
    if layout == "bsr_batch":
        batch, n_rows, n_cols = size
        br, bc = blocks
        n_row_blocks = n_rows // br
        n_col_blocks = n_cols // bc
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


def _torch_col_indices_copy(inp):
    # torch.ops.aten.col_indices_copy is registered as
    # CompositeExplicitAutogradNonFunctional; some builds restrict its
    # dispatch-key set to dense backends and raise NotImplementedError on
    # sparse tensors. Benchmark the operator's exact native body --
    # col_indices(self).clone(contiguous) -- which shares call semantics with
    # the candidate on every build.
    try:
        return torch.ops.aten.col_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.col_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


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


class ColIndicesCopyBenchmark(base.GenericBenchmark):
    # col_indices_copy is a sparse metadata accessor; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated (layout, size,
    # nnz, blocks) cases instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COLS


@pytest.mark.col_indices_copy
def test_col_indices_copy():
    bench = ColIndicesCopyBenchmark(
        op_name="col_indices_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=_torch_col_indices_copy,
        gems_op=getattr(flag_gems, "col_indices_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
