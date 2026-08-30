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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
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
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts  # noqa: E402

# aten::_dimV(Tensor self) -> int reports the dense dimension count of a
# sparse tensor. It is a pure metadata query (the measured work is dispatch and
# layout introspection, never data movement), and dense / SparseCsr tensors
# raise NotImplementedError for it, so every benchmark input is a sparse COO
# tensor. The shapes below cover representative logical sizes across ranks 2-4;
# the actual device allocation stays tiny because nnz is fixed and small.
_DIMV_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz
# only affects input allocation, not the measured call.
_DIMV_NNZ = 1024


def _make_sparse_input(shape, dense_dim, dtype, device, nnz=_DIMV_NNZ, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_dim = len(shape) - dense_dim
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
    # Cover all-sparse 2-D layouts (dense_dim == 0) and mixed sparse+dense
    # 3-D/4-D layouts (dense_dim == 2); every derived dense_dim stays within
    # [0, ndim] so additional shapes merged in by the comprehensive bench level
    # remain valid.
    dense_dim = 0 if len(shape) <= 2 else 2
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dense_dim": dense_dim},
        builder_args=(shape, dense_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dense_dim = plan.builder_args
    inp = _make_sparse_input(shape, dense_dim, dtype, device)
    return inp, {}


class DimVBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO tensors."""

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
