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
from _pytest.mark.structures import Mark, MarkDecorator

from . import base, consts, utils

# ``_version`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._version`` and ``-m _version`` both
# work.
setattr(
    pytest.mark,
    "_version",
    MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_version(Tensor self) -> int reads the per-tensor version counter
# (bumped by every in-place mutation). It is a pure O(1) metadata query whose
# measured latency is independent of the tensor data, so the benchmark shapes
# only control the input allocation outside the timed region; they cover ranks
# 1-4 with representative element counts.
_VERSION_SHAPES = [
    (1,),
    (1024,),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (16, 256, 256, 16),
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


class VersionBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark with shapes tuned for the O(1) _version query."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _VERSION_SHAPES


@pytest.mark._version
def test__version():
    bench = VersionBenchmark(
        op_name="_version",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._version,
        # ``flag_gems._version`` is the package version module (package
        # metadata), not the operator callable, so the candidate is resolved
        # from the process-local override injected by KernelGen; without one
        # the benchmark falls back to the dispatcher route.
        gems_op=None,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
