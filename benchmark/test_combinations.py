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

# The KernelGen harness runs pytest in-process with ``--import-mode=importlib``,
# which does not prepend the checkout root to sys.path, so the ``benchmark``
# package may resolve to the harness's own package or not resolve at all.
# Re-point it at this file's directory before importing the benchmark helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts, utils  # noqa: E402

# aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
# materializes all length-r combinations of a 1-D input: C(n, r) rows without
# replacement and C(n + r - 1, r) rows with replacement. The output grows
# combinatorially (the r=2 output of a 4096-element input already has ~8.4M
# rows), so the case list keeps n modest while still spanning small, medium and
# large (n, r) pairs for both replacement modes. Each case is a (n, r,
# with_replacement) triple; the default r=2 / without-replacement workload is
# the primary one.
_COMBINATIONS_CASES = [
    (64, 2, False),
    (256, 2, False),
    (1024, 2, False),
    (4096, 2, False),
    (64, 3, False),
    (256, 3, False),
    (64, 3, True),
]


def _case_fn(shape, dtype):
    del dtype
    n, r, with_replacement = shape
    yield base.BenchmarkCasePlan(
        shape={"input": (n,)},
        params={"r": r, "with_replacement": with_replacement},
        builder_args=(n, r, with_replacement),
    )


def _build_inputs_fn(plan, dtype, device):
    n, r, with_replacement = plan.builder_args
    inp = utils.generate_tensor_input((n,), dtype, device)
    return inp, {"r": r, "with_replacement": with_replacement}


class CombinationsBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark for the combinatorial gather of combinations.

    The default shape set contains multi-dimensional shapes, which
    aten::combinations rejects at runtime, so the case list is restricted to the
    1-D (n, r, with_replacement) triples above.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = _COMBINATIONS_CASES


@pytest.mark.combinations
def test_combinations():
    bench = CombinationsBenchmark(
        op_name="combinations",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.combinations,
        # KernelGen injects the candidate via override_gems_op(); the default
        # module callable may not exist until the op is merged into FlagGems.
        gems_op=getattr(flag_gems, "combinations", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
