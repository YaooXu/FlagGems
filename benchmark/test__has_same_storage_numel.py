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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# ``_has_same_storage_numel`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._has_same_storage_numel`` and
# ``-m _has_same_storage_numel`` both work.
setattr(
    pytest.mark,
    "_has_same_storage_numel",
    MarkDecorator(
        Mark("_has_same_storage_numel", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Make sure the FlagGems checkout that physically contains this file is the one
# used for the sibling ``benchmark`` package. Under pytest
# ``--import-mode=importlib`` the process sys.path may hold an unrelated entry
# that shadows this checkout's ``benchmark`` package; insert the checkout root
# at the front and re-import the package from this file's own directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import benchmark as _bench_pkg  # noqa: E402

if _HERE not in getattr(_bench_pkg, "__path__", []):
    sys.modules.pop("benchmark", None)
    import benchmark as _bench_pkg  # noqa: E402

from . import base, consts, utils  # noqa: E402

# aten::_has_same_storage_numel is a pure storage-metadata query: it compares
# self.storage().numel() with other.storage().numel() and allocates nothing, so
# the benchmark measures dispatch overhead. The default shape set contains a
# 1-B-element 1-D tensor whose cost would be dominated by input allocation; use
# allocation-friendly shapes instead.
_HAS_SAME_STORAGE_NUMEL_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 256),
    (20, 320, 15),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "other": shape},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    self_inp = utils.generate_tensor_input(shape, dtype, device)
    other_inp = utils.generate_tensor_input(shape, dtype, device)
    return self_inp, other_inp


class HasSameStorageNumelBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _HAS_SAME_STORAGE_NUMEL_SHAPES


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel():
    bench = HasSameStorageNumelBenchmark(
        op_name="_has_same_storage_numel",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._has_same_storage_numel,
        gems_op=getattr(flag_gems, "_has_same_storage_numel", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
