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

import math
import os
import sys

import pytest
import torch

import flag_gems

# Same package-resolution bootstrap as the correctness suite: the benchmark
# package that ships with this file must win over any other top-level
# ``benchmark`` package already importable on sys.path (the KernelGen harness
# runs pytest in-process with its own ``benchmark`` package).
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_BENCH_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
_IMPORTED_BENCHMARK = sys.modules.get("benchmark")
if _IMPORTED_BENCHMARK is not None and os.path.abspath(
    getattr(_IMPORTED_BENCHMARK, "__file__", "")
) != os.path.join(_BENCH_DIR, "__init__.py"):
    del sys.modules["benchmark"]

from . import base, consts, utils  # noqa: E402

# aten::atleast_1d is a pure view/identity op (0-dim -> (1,) view, ndim >= 1
# returned as-is). No public Benchmark family models a view identity op, so both
# overloads use a two-phase GenericBenchmark (case_fn + build_inputs_fn). The
# 0-dim scalar case is the defining workload and is prepended to the shape set.
# gems_op is resolved through getattr because flag_gems.atleast_1d is not yet
# registered as a direct callable; KernelGen's override_gems_op("atleast_1d",
# ...) still wins at run time via flag_gems.testing.resolve_gems_op.


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


def _sequence_case_fn(shape, dtype):
    del dtype
    # Mix a 0-dim scalar with the current benchmark shape so the sequence
    # overload exercises both the scalar -> (1,) view and the identity path.
    seq_shapes = [(), shape, shape]
    yield base.BenchmarkCasePlan(
        shape={"input": seq_shapes},
        params={},
        builder_args=(seq_shapes,),
    )


def _sequence_build_inputs_fn(plan, dtype, device):
    seq_shapes = plan.builder_args[0]
    inp = [utils.generate_tensor_input(s, dtype, device) for s in seq_shapes]
    return inp, {}


class Atleast1DBenchmark(base.GenericBenchmark):
    # A view op's latency is dominated by the call overhead, not the tensor
    # size, so capping the input numel avoids allocating multi-GB inputs (the
    # generic DEFAULT_SHAPES include 1G-element tensors) for no signal.
    MAX_NUMEL = 2**24  # 16M elements

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_NUMEL
        ]
        # atleast_1d's defining case is the 0-dim scalar -> (1,) view.
        if () not in self.shapes:
            self.shapes = [()] + list(self.shapes)


@pytest.mark.atleast_1d
def test_atleast_1d():
    bench = Atleast1DBenchmark(
        op_name="atleast_1d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_1d,
        gems_op=getattr(flag_gems, "atleast_1d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atleast_1d_sequence
def test_atleast_1d_sequence():
    bench = Atleast1DBenchmark(
        op_name="atleast_1d",
        case_fn=_sequence_case_fn,
        build_inputs_fn=_sequence_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_1d.Sequence,
        gems_op=getattr(flag_gems, "atleast_1d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
