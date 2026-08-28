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

# The KernelGen verification harness stages these files in an isolated copy
# of the FlagGems tree whose parent directory is not on sys.path. Make the
# ``tests``/``benchmark`` packages importable regardless of the harness
# process's sys.path so the relative imports below resolve.
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts, utils  # noqa: E402

# ``_unpack_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._unpack_dual`` and ``-m _unpack_dual``
# both work.
setattr(
    pytest.mark,
    "_unpack_dual",
    MarkDecorator(Mark("_unpack_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor
# tangent) is the forward-mode AD dual-construction inverse: it reads the primal
# (as an aliasing view) and the tangent off a dual tensor created at an active
# forward-mode AD level. The level the op names must be live when it is called,
# and forward-mode AD does not support nested dual_level() contexts, so the
# whole benchmark runs inside a single ``dual_level()`` and the level index it
# assigns is threaded through the input builder to both the reference
# (torch_op) and the candidate (gems_op). The candidate kernel only consumes
# dual metadata (the tangent is a separate tensor), so each case materializes
# primal + tangent, creates the dual view, and reads both once.


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": "forward-ad-active"},
        builder_args=(shape,),
    )


class UnpackDualBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark that supplies a dual tensor and its level.

    aten::_unpack_dual needs a level alongside the dual tensor, which the
    pointwise families do not supply, so the case builder and input builder
    forward it explicitly. The default shape set contains a 1G-element core
    shape whose fp32 primal + tangent would need ~8 GiB and OOM on busy GPUs;
    cap the shapes while keeping performance-relevant sizes (2**26 elements =
    256 MiB fp32 per tensor).
    """

    MAX_ELEMENTS = 2**26

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark._unpack_dual
def test__unpack_dual():
    # The level must stay live for the entire benchmark run, so dual_level() is
    # entered here and the input builder closes over the exact level index it
    # assigns (a hardcoded level would break on torch builds whose forward AD
    # level counter is not zero-based).
    with torch.autograd.forward_ad.dual_level() as level:

        def _build_inputs_fn(plan, dtype, device):
            shape = plan.builder_args[0]
            primal = utils.generate_tensor_input(shape, dtype, device)
            tangent = utils.generate_tensor_input(shape, dtype, device)
            dual = torch.ops.aten._make_dual(primal, tangent, level)
            return dual, level

        bench = UnpackDualBenchmark(
            op_name="_unpack_dual",
            case_fn=_case_fn,
            build_inputs_fn=_build_inputs_fn,
            torch_op=torch.ops.aten._unpack_dual,
            gems_op=getattr(flag_gems, "_unpack_dual", None),
            dtypes=consts.FLOAT_DTYPES,
        )
        bench.run()
