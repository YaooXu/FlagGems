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

import sys
from pathlib import Path

import pytest
import torch

import flag_gems


def _unshadow_packages():
    """Realign sys.path / sys.modules so the ``tests`` and ``benchmark`` packages
    that physically contain this file win over same-named packages on the process
    sys.path.

    The KernelGen TestWriter harness stages a temporary copy of the FlagGems tree
    and runs ``pytest --import-mode=importlib`` from the kernelgen repo process,
    whose sys.path contains the kernelgen root (which ships its own ``tests/``
    package). Leaving the real package root at ``sys.path[0]`` makes both the
    relative imports below and the benchmark conftest load resolve to the real
    packages.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    for name in ("tests", "benchmark"):
        real = pkg_root / name
        if not real.is_dir():
            continue
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None):
            try:
                if Path(mod.__file__).resolve().parent == real.resolve():
                    continue  # already the real package
            except Exception:
                pass
        prefix = name + "."
        for key in [
            k
            for k in sys.modules
            if (k == name or k.startswith(prefix)) and k != __name__
        ]:
            sys.modules.pop(key, None)
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))


_unshadow_packages()

from . import base, consts  # noqa: E402

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
    # coalesce is a sparse op; there are no meaningful dense shapes in
    # core_shapes.yaml, so benchmark dedicated (shape, nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COALESCE_SHAPES


@pytest.mark.coalesce
def test_coalesce():
    bench = CoalesceBenchmark(
        op_name="coalesce",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.coalesce,
        gems_op=getattr(flag_gems, "coalesce", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
