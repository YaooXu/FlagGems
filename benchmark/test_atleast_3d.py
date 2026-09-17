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

import pytest
import torch

import flag_gems

from . import base, consts, utils

# aten::atleast_3d is a pure view/identity op (0-dim -> (1, 1, 1), 1-dim ->
# (1, N, 1), 2-dim -> (M, N, 1); ndim >= 3 returned unchanged) with two
# overloads (Tensor and Tensor[]). No public Benchmark family models a view
# op, so both overloads use the two-phase GenericBenchmark (case_fn +
# build_inputs_fn), never a bare legacy input_fn.
#
# A view's latency is dominated by dispatch/call overhead rather than tensor
# size, so the shape set is curated (one case per rank 0..4 plus a large 2-D
# and 3-D case) and stays small: the generic DEFAULT_SHAPES include 1G-element
# tensors that would only burn memory for no signal.
#
# gems_op is resolved inside each test function (never at import time) so the
# process-local override installed by KernelGen for this run wins; when no
# candidate is registered flag_gems.testing.resolve_gems_op raises LookupError
# and the benchmark falls back to its normal torch_op reference route.

_CURATED_SHAPES = [
    (),  # 0-dim scalar -> (1, 1, 1)
    (1,),  # single-element 1-dim -> (1, 1, 1)
    (256,),  # regular 1-dim -> (1, 256, 1)
    (1024, 1024),  # large 2-dim -> (1024, 1024, 1)
    (20, 320, 15),  # 3-dim identity
    (16, 128, 64),  # 3-dim identity
    (8, 16, 32, 4),  # 4-dim identity
]


class Atleast3DBenchmark(base.GenericBenchmark):
    # Curated, size-bounded shape set: a view op has no data-dependent work, so
    # the larger generic shapes add allocation time without changing the
    # measured dispatch cost.
    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        self.shapes = list(_CURATED_SHAPES)
        self.shape_desc = "rank-complete view shapes"


def _case_fn(shape, dtype):
    # One Workload per shape for the Tensor overload.
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
    # One Workload per shape for the Tensor[] overload, mixing a 0-dim scalar,
    # a 1-dim tensor, a 2-dim tensor and the current shape so the scalar ->
    # (1,1,1), 1-dim -> (1,N,1), 2-dim -> (M,N,1) and >= 3-dim identity paths
    # are all timed.
    del dtype
    seq_shapes = [(), (3,), (4, 5), shape]
    yield base.BenchmarkCasePlan(
        shape={"input": seq_shapes},
        params={},
        builder_args=(seq_shapes,),
    )


def _sequence_build_inputs_fn(plan, dtype, device):
    seq_shapes = plan.builder_args[0]
    inp = [utils.generate_tensor_input(s, dtype, device) for s in seq_shapes]
    return inp, {}


def _resolve_named_gems_op(name):
    # Resolution order: (1) the override installed by KernelGen, (2) the direct
    # flag_gems callable, (3) None -> the benchmark keeps its torch_op
    # reference. Never resolved at import time.
    default = getattr(flag_gems, name.replace(".", "_"), None)
    if default is None:
        default = getattr(flag_gems, name, None)
    try:
        return flag_gems.testing.resolve_gems_op(name, default)
    except LookupError:
        return None


def _resolve_gems_op():
    return _resolve_named_gems_op("atleast_3d")


def _resolve_gems_op_sequence():
    # Accept a dedicated Sequence override or a single callable handling both
    # overloads.
    for name in ("atleast_3d.Sequence", "atleast_3d_sequence", "atleast_3d"):
        op = _resolve_named_gems_op(name)
        if op is not None:
            return op
    return None


@pytest.mark.atleast_3d
@pytest.mark.atleast_3d_benchmark
def test_atleast_3d():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d,
        gems_op=_resolve_gems_op(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atleast_3d_sequence
@pytest.mark.atleast_3d_benchmark
def test_atleast_3d_sequence():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_sequence_case_fn,
        build_inputs_fn=_sequence_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d.Sequence,
        gems_op=_resolve_gems_op_sequence(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
