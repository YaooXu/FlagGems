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
    # Mix a 0-dim scalar, a 1-dim tensor, a 2-dim tensor and the current
    # benchmark shape so the sequence overload exercises the scalar ->
    # (1, 1, 1), 1-dim -> (1, N, 1), 2-dim -> (M, N, 1) and >= 3-dim identity
    # paths.
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


class Atleast3DBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # atleast_3d's defining cases are the 0-dim scalar -> (1, 1, 1),
        # 1-dim -> (1, N, 1) and 2-dim -> (M, N, 1) views.
        if () not in self.shapes:
            self.shapes = [()] + list(self.shapes)
        if (3,) not in self.shapes:
            self.shapes = [(3,)] + list(self.shapes)
        if (4, 5) not in self.shapes:
            self.shapes = [(4, 5)] + list(self.shapes)


@pytest.mark.atleast_3d
def test_atleast_3d():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d,
        gems_op=getattr(flag_gems, "atleast_3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atleast_3d_sequence
def test_atleast_3d_sequence():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_sequence_case_fn,
        build_inputs_fn=_sequence_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d.Sequence,
        gems_op=getattr(flag_gems, "atleast_3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
