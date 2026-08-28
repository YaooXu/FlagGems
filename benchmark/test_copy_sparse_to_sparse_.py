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

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import base, consts`` cannot resolve this checkout's benchmark
# package through normal package discovery. Put the checkout root on sys.path
# and re-point the ``benchmark`` package at THIS checkout (belt-and-suspenders:
# the correctness file already does this when it runs first, but this keeps the
# benchmark file self-contained).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts  # noqa: E402

# aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False)
# -> Tensor(a!) is an in-place sparse-COO-to-sparse-COO copy whose cost scales
# with nnz (and the dense trailing dims of hybrid layouts), so each case below
# is a (logical_shape, sparse_dim, nnz) triple. There are no meaningful dense
# shapes in core_shapes.yaml for this op, so dedicated sparse cases are used.
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
    # Sparse COO input built directly on the benchmark device. Indices are
    # drawn with replacement (a valid, possibly uncoalesced sparse structure);
    # the copy cost depends only on nnz and the dense trailing dims.
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in shape[:sparse_dim]
        ]
    )
    values = torch.randn(values_shape, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(shape, dtype):
    del dtype
    logical_shape, sparse_dim, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": logical_shape},
        params={"nnz": nnz},
        builder_args=(logical_shape, sparse_dim, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    logical_shape, sparse_dim, nnz = plan.builder_args
    src = _make_sparse_input(logical_shape, sparse_dim, nnz, dtype, device)
    dst = torch.zeros_like(src)
    # In-place copy: the harness measures op(dst, src, non_blocking=False)
    # repeatedly; each iteration copies src into dst again, so timing is stable.
    return dst, src, {"non_blocking": False}


class CopySparseToSparseBenchmark(base.GenericBenchmark):
    # copy_sparse_to_sparse_ is a sparse-to-sparse copy; the dense default
    # shapes in core_shapes.yaml do not apply, so benchmark dedicated
    # (logical_shape, sparse_dim, nnz) triples instead.
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
