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

# aten::sparse_compressed_tensor constructs a sparse compressed tensor
# (CSR/CSC/BSR/BSC) from compressed_indices, plain_indices, and values. The
# generic factory needs layout= (mandatory), and on GPU both dtype= and device=
# must be passed explicitly (the aten op does not infer dtype from values and
# creates the instance on CPU when device is omitted). The construction cost is
# dominated by validation and index/value allocation, so the benchmarks below
# use (layout, sparse shape, nnz) triples with performance-relevant nnz instead
# of the dense core_shapes.yaml set.
_SPARSE_COMPRESSED_SHAPES = [
    (torch.sparse_csr, (1024, 1024), 65536),
    (torch.sparse_csr, (1024, 1024), 262144),
    (torch.sparse_csc, (1024, 1024), 262144),
    (torch.sparse_csr, (4096, 4096), 1048576),
    (torch.sparse_bsr, (2048, 2048), 262144),
    (torch.sparse_csr, (256, 256, 256), 1048576),
]

_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2


def _make_input(layout, shape, nnz, dtype, device):
    # Random but always-valid compressed sparse structure, generated directly
    # on the benchmark device. Block layouts get (nnz, block, block) value
    # tensors.
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    entries = batch + (nnz,)
    comp = torch.randint(0, comp_dim, entries, dtype=torch.long, device=device)
    plain = torch.randint(0, plain_dim, entries, dtype=torch.long, device=device)
    order = torch.argsort(comp * plain_dim + plain, dim=-1)
    comp = torch.gather(comp, -1, order)
    plain = torch.gather(plain, -1, order)
    counts = torch.stack(
        [
            torch.bincount(comp[idx], minlength=comp_dim)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (comp_dim,))
    compressed = torch.zeros(batch + (comp_dim + 1,), dtype=torch.long, device=device)
    compressed[..., 1:] = torch.cumsum(counts, -1)
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    values = torch.randn(entries + block_shape, dtype=dtype, device=device)
    return compressed, plain, values


def _case_fn(shape, dtype):
    del dtype
    layout, shape_, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape_},
        params={"nnz": nnz, "layout": layout},
        builder_args=(layout, shape_, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, shape, nnz = plan.builder_args
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, device)
    # The kwargs dict travels at the top level of the returned tuple so
    # unpack_to_args_kwargs places the tensors in args and the dict in kwargs;
    # layout/device/dtype must go through the dict (they are neither tensors
    # nor plain scalars).
    return (
        compressed,
        plain,
        values,
        {
            "size": list(shape),
            "layout": layout,
            "dtype": dtype,
            "device": device,
        },
    )


class SparseCompressedTensorBenchmark(base.GenericBenchmark):
    # Sparse constructor; there are no meaningful dense shapes in
    # core_shapes.yaml, so benchmark dedicated (layout, shape, nnz) triples.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _SPARSE_COMPRESSED_SHAPES


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor():
    bench = SparseCompressedTensorBenchmark(
        op_name="sparse_compressed_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_compressed_tensor,
        gems_op=getattr(flag_gems, "sparse_compressed_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
