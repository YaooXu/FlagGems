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

# KernelGen's in-process verification stages the test files into an isolated
# temp copy of the checkout, where the relative ``from . import base`` cannot
# resolve this checkout's benchmark package through normal package discovery.
# Put the checkout root on sys.path so the ``benchmark`` package resolves to
# THIS checkout no matter how pytest is invoked (belt-and-suspenders: the
# correctness file already does this when it runs first, but this keeps the
# benchmark file self-contained).
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base  # noqa: E402

# aten::can_cast(ScalarType from_, ScalarType to) -> bool is a pure dtype-
# metadata query: it allocates nothing and never touches the device, so the
# benchmark measures pure dispatch/function-call overhead. There are no tensor
# inputs to materialize; the only meaningful case dimension is the (from_, to)
# dtype pair. The representative pairs below cover every scalar-type family in
# both directions, including both True (same-family widening, bool -> float,
# complex widening) and False (float -> int, float -> bool, int -> float16)
# outcomes. No public Benchmark family covers a tensor-free dtype query, so the
# two-phase GenericBenchmark drives the case list directly from _case_fn /
# _build_inputs_fn; the candidate is passed explicitly via gems_op and the perf
# reference is the aten op itself (torch.ops.aten.can_cast), both called with
# the same two-dtype signature.
_CAN_CAST_BENCH_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
    (torch.float32, torch.float64),
    (torch.int32, torch.int64),
    (torch.int64, torch.int32),
    (torch.int32, torch.float32),
    (torch.float32, torch.int32),
    (torch.float64, torch.int64),
    (torch.bool, torch.float32),
    (torch.float32, torch.bool),
    (torch.complex64, torch.complex128),
    (torch.int16, torch.float16),
]

# The op never creates tensors, so the benchmark dtype bucket is irrelevant; a
# single bucket keeps the case list free of redundant per-dtype repeats.
_CAN_CAST_BENCH_DTYPES = [torch.float32]


def _case_fn(shape, dtype):
    del shape, dtype
    for from_dtype, to_dtype in _CAN_CAST_BENCH_PAIRS:
        yield base.BenchmarkCasePlan(
            shape={"from_": str(from_dtype), "to": str(to_dtype)},
            params={"from_": str(from_dtype), "to": str(to_dtype)},
            builder_args=(from_dtype, to_dtype),
        )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    from_dtype, to_dtype = plan.builder_args
    return from_dtype, to_dtype


class CanCastBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark for the tensor-free dtype-pair op can_cast."""

    def set_shapes(self, shape_file_path=None):
        # There is no tensor data for can_cast; the case list is driven solely
        # by the (from_, to) dtype pairs enumerated in _case_fn.
        self.shapes = [(0,)]

    def set_more_shapes(self):
        # Additional tensor shapes are meaningless for a dtype-metadata query.
        return []


@pytest.mark.can_cast
def test_can_cast():
    bench = CanCastBenchmark(
        op_name="can_cast",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.can_cast,
        gems_op=getattr(flag_gems, "can_cast", None),
        dtypes=_CAN_CAST_BENCH_DTYPES,
    )
    bench.run()
