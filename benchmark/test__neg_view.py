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

# The KernelGen verification harness stages these files in an isolated copy of
# the FlagGems tree whose parent directory is not on sys.path. Make the
# ``tests``/``benchmark`` packages importable regardless of the harness
# process's sys.path so the relative imports below resolve.
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts, utils  # noqa: E402

# ``_neg_view`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._neg_view`` and ``-m _neg_view`` both
# work.
setattr(
    pytest.mark,
    "_neg_view",
    MarkDecorator(Mark("_neg_view", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_neg_view is a zero-copy negative view: it shares the input storage and
# only toggles the lazy neg bit, so the benchmark measures dispatch and
# view-construction overhead. The default shape set contains a 1-B-element 1-D
# tensor whose cost would be dominated by input allocation; use
# allocation-friendly shapes instead.
NEG_VIEW_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
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


class NegViewBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = NEG_VIEW_SHAPES


@pytest.mark._neg_view
def test__neg_view():
    bench = NegViewBenchmark(
        op_name="_neg_view",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._neg_view,
        gems_op=getattr(flag_gems, "_neg_view", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
