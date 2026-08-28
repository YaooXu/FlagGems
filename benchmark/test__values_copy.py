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

# The KernelGen verification harness stages this file in a temporary tree and
# runs pytest in-process with --import-mode=importlib, where the checkout root
# is not on sys.path (a `python -m pytest` invocation would normally place it
# there). Insert the checkout root so the sibling benchmark package (base,
# consts, conftest) below resolves identically in the in-tree and staged
# verification layouts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# ``_values_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._values_copy`` and ``-m
# _values_copy`` both work.
setattr(
    pytest.mark,
    "_values_copy",
    MarkDecorator(Mark("_values_copy", (), {}, _ispytest=True), _ispytest=True),
)

# (sparse_shape, dense_shape, nnz). _values_copy materializes the
# (nnz,) + dense_shape values tensor of a sparse COO tensor as a fresh
# contiguous copy — a data-movement accessor whose cost is proportional to
# nnz * prod(dense_shape) (the size of the returned values tensor), so
# benchmark a spread of sparse ranks, dense ranks and nnz values. The
# device-side allocation stays small relative to the logical size because only
# nnz entries are stored.
_VALUES_COPY_SHAPES = [
    ((1024, 1024), (), 65536),
    ((1024, 1024), (), 1048576),
    ((1024, 1024), (16,), 262144),
    ((256, 256, 256), (), 1048576),
    ((128, 128, 128, 128), (8,), 1048576),
    ((4096, 4096), (64,), 1048576),
]


def _torch_values_copy(inp):
    # torch.ops.aten._values_copy is registered as
    # CompositeExplicitAutogradNonFunctional, whose dispatch-key set excludes
    # the Sparse functionality key, so it is unreachable on sparse tensors in
    # the installed build. Benchmark the operator's exact native body —
    # _values(self).clone(contiguous) — which shares call semantics with the
    # candidate.
    return torch.ops.aten._values(inp).clone(memory_format=torch.contiguous_format)


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


class ValuesCopyBenchmark(base.GenericBenchmark):
    # _values_copy is a sparse data-movement accessor; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated (sparse_shape,
    # dense_shape, nnz) triples instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _VALUES_COPY_SHAPES


@pytest.mark._values_copy
def test__values_copy():
    bench = ValuesCopyBenchmark(
        op_name="_values_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=_torch_values_copy,
        gems_op=getattr(flag_gems, "_values_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
