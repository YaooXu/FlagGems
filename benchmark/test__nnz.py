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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# ``_nnz`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._nnz`` and ``-m _nnz`` both work.
setattr(
    pytest.mark,
    "_nnz",
    MarkDecorator(Mark("_nnz", (), {}, _ispytest=True), _ispytest=True),
)

# Make sure the FlagGems checkout that physically contains this file is the one
# used for the sibling ``benchmark`` package. Under pytest
# ``--import-mode=importlib`` the process sys.path may hold an unrelated entry
# that shadows this checkout's ``benchmark`` package; insert the checkout root
# at the front and re-import the package from this file's own directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg

from . import base, consts  # noqa: E402

# aten::_nnz(Tensor self) -> int reports the number of stored entries of a
# sparse tensor. It is a pure metadata query (the measured work is dispatch and
# layout introspection, never data movement), and dense tensors raise
# NotImplementedError for it, so every benchmark input is a sparse COO tensor.
# The shapes below cover representative logical sizes across ranks 2-4; the
# actual device allocation stays tiny because nnz is fixed and small.
_NNZ_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz
# only affects input allocation, not the measured call.
_NNZ = 1024


def _make_sparse_input(shape, sparse_dim, dtype, device, nnz=_NNZ, seed=0):
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
        params={"sparse_dim": sparse_dim, "nnz": _NNZ},
        builder_args=(shape, sparse_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim = plan.builder_args
    inp = _make_sparse_input(shape, sparse_dim, dtype, device)
    return inp, {}


class NnzBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO tensors."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _NNZ_SHAPES


@pytest.mark._nnz
def test__nnz():
    bench = NnzBenchmark(
        op_name="_nnz",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nnz,
        gems_op=getattr(flag_gems, "_nnz", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
