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

import sys as _sys
from pathlib import Path as _Path

# pytest --import-mode=importlib imports this module as <pkg>.test__coalesce,
# where <pkg> is the "tests" or "benchmark" package of the checkout that
# actually holds this file (the KernelGen verification harness stages a temp
# copy of the FlagGems tree). When the driving process also has a same-named
# package on sys.path (e.g. the KernelGen repo's own tests/ directory), a bare
# relative import below would bind to that foreign package instead. Put the
# checkout root of *this* file first in sys.path so the relative imports
# resolve to the support files (accuracy_utils/test_utils/base/consts) that
# ship next to it.
_CHECKOUT_ROOT = _Path(__file__).resolve().parent.parent
if str(_CHECKOUT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CHECKOUT_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# ``_coalesce`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._coalesce`` and ``-m _coalesce`` both
# work.
setattr(
    pytest.mark,
    "_coalesce",
    MarkDecorator(Mark("_coalesce", (), {}, _ispytest=True), _ispytest=True),
)

# (sparse shape, nnz). Coalescing work scales with nnz, and drawing nnz
# entries over the index space guarantees duplicate indices (real merging
# work) while keeping the tensors small enough for repeated benchmarking.
_COALESCE_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((2048, 2048), 2097152),
    ((256, 256, 256), 1048576),
]


def _case_fn(shape, dtype):
    del dtype
    shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"nnz": nnz},
        builder_args=(shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, nnz = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return inp, {}


class CoalesceBenchmark(base.GenericBenchmark):
    # _coalesce is a sparse op; there are no meaningful dense shapes in
    # core_shapes.yaml, so benchmark dedicated (shape, nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COALESCE_SHAPES


@pytest.mark._coalesce
def test__coalesce():
    bench = CoalesceBenchmark(
        op_name="_coalesce",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._coalesce,
        gems_op=getattr(flag_gems, "_coalesce", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
