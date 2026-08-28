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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts

# ``_dimI`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimI`` and ``-m _dimI`` both work.
setattr(
    pytest.mark,
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimI(Tensor self) -> int reports the sparse dimension count of a
# sparse tensor. It is a pure metadata query (the measured work is dispatch and
# layout introspection, never data movement), and dense tensors raise
# NotImplementedError for it, so every benchmark input is a sparse COO tensor.
# The shapes below cover representative logical sizes across ranks 2-4; the
# actual device allocation stays tiny because nnz is fixed and small.
_DIMI_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz
# only affects input allocation, not the measured call.
_DIMI_NNZ = 1024


def _make_sparse_input(shape, sparse_dim, dtype, device, nnz=_DIMI_NNZ, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    if dtype.is_floating_point:
        values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(shape, dtype):
    del dtype
    # Cover all-sparse (2-D) and mixed sparse+dense layouts (3-D/4-D); every
    # derived sparse_dim stays within [1, ndim] so additional shapes merged in
    # by the comprehensive bench level remain valid.
    sparse_dim = len(shape) if len(shape) <= 2 else len(shape) - 1
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"sparse_dim": sparse_dim},
        builder_args=(shape, sparse_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim = plan.builder_args
    inp = _make_sparse_input(shape, sparse_dim, dtype, device)
    return inp, {}


class DimIBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO tensors."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _DIMI_SHAPES


@pytest.mark._dimI
def test__dimI():
    bench = DimIBenchmark(
        op_name="_dimI",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._dimI,
        gems_op=getattr(flag_gems, "_dimI", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
