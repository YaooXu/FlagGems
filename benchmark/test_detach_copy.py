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

# aten::detach_copy(Tensor self) -> Tensor is a pure memory copy that returns a
# new contiguous tensor (no autograd, no aliasing), so the benchmark shapes are
# chosen to exercise copy bandwidth at several sizes. The default shape set
# contains a 1 GiB-element tensor, which would need ~4 GB per fp32 input and is
# impractical for repeated copy runs; the shape list below is therefore
# restricted to the memory-copy-relevant sizes.
DETACH_COPY_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (256, 1024, 1024),
    (512, 512, 512),
]


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


class DetachCopyBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark for the detach_copy memory-copy workload."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = DETACH_COPY_SHAPES


@pytest.mark.detach_copy
def test_detach_copy():
    bench = DetachCopyBenchmark(
        op_name="detach_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.detach_copy,
        gems_op=getattr(flag_gems, "detach_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
