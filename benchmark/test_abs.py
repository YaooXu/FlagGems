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

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import base, consts`` cannot resolve this checkout's benchmark
# package through normal package discovery. Put the checkout root on sys.path so
# the ``benchmark`` package resolves to THIS checkout no matter how pytest is
# invoked (belt-and-suspenders: the correctness file already does this when it
# runs first, but this keeps the benchmark file self-contained).
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# abs is a unary pointwise op, so the public UnaryPointwiseBenchmark family
# covers its semantics (input shapes from core_shapes.yaml). The candidate is
# passed explicitly via gems_op and the perf reference is the aten op itself
# (torch.ops.aten.abs / abs_); both are called with the same single-tensor
# signature that the family's build_inputs produces.


@pytest.mark.abs
def test_abs():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs",
        torch_op=torch.ops.aten.abs,
        gems_op=flag_gems.abs,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.abs_
def test_abs_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs_",
        torch_op=torch.ops.aten.abs_,
        gems_op=flag_gems.abs_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
