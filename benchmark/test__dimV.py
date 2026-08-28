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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate
# a marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
)

# (sparse_shape, dense_shape, nnz). _dimV is a sparse metadata query whose
# result is ``len(dense_shape)``; benchmark a spread of sparse ranks, dense
# ranks and nnz values so candidate strategies see a realistic workload.
# The device-side allocation stays small relative to the logical size because
# only nnz entries are stored.
_DIMV_SHAPES = [
    ((1024, 1024), (), 65536),
    ((1024, 1024), (32,), 262144),
    ((256, 256, 256), (16,), 1048576),
    ((1024, 1024), (16, 16), 1048576),
    ((128, 128, 128, 128), (8,), 1048576),
    ((4096, 4096), (64,), 1048576),
]


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, dense_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape + dense_shape},
        params={"nnz": nnz},
        builder_args=(sparse_shape, dense_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, dense_shape, nnz = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in sparse_shape
        ]
    )
    values_shape = (nnz,) + tuple(dense_shape)
    values = torch.randn(values_shape, dtype=dtype, device=device)
    size = tuple(sparse_shape) + tuple(dense_shape)
    inp = torch.sparse_coo_tensor(indices, values, size, device=device)
    return inp, {}


class DimVBenchmark(base.GenericBenchmark):
    # _dimV is a sparse op; there are no meaningful dense shapes in
    # core_shapes.yaml, so benchmark dedicated (sparse_shape, dense_shape, nnz)
    # triples instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _DIMV_SHAPES


@pytest.mark._dimV
def test__dimV():
    bench = DimVBenchmark(
        op_name="_dimV",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._dimV,
        gems_op=getattr(flag_gems, "_dimV", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
