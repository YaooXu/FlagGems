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

# aten::col_indices(Tensor(a) self) -> Tensor(a) returns the batch_dims +
# (nnz,) int64 column index tensor of a sparse CSR tensor -- a metadata
# accessor whose result is an alias of the input's internal col_indices
# storage. Its cost is proportional to nnz (the size of the returned tensor)
# and independent of the stored values, so benchmark a spread of logical
# shapes (nrows, ncols) / (batch, nrows, ncols) / (batch_1, batch_2, nrows,
# ncols) and nnz values. The device-side allocation stays small relative to
# the logical size because only nnz entries are stored.
_COL_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((16, 1024, 1024), 262144),
    ((8, 256, 256), 1048576),
    ((64, 1024, 1024), 131072),
    ((4, 8, 256, 256), 262144),
]


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


class ColIndicesBenchmark(base.GenericBenchmark):
    # col_indices is a sparse metadata accessor; there are no meaningful dense
    # shapes in core_shapes.yaml, so benchmark dedicated (logical_shape, nnz)
    # pairs instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COL_SHAPES


@pytest.mark.col_indices
def test_col_indices():
    bench = ColIndicesBenchmark(
        op_name="col_indices",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.col_indices,
        gems_op=getattr(flag_gems, "col_indices", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
