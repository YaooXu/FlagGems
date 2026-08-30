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

# ``_new_zeros_with_same_feature_meta`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so ``-m
# _new_zeros_with_same_feature_meta`` works.
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta", (), {}, _ispytest=True),
        _ispytest=True,
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

# aten::_new_zeros_with_same_feature_meta allocates a zero tensor whose shape
# is ``self.shape[:self_num_batch_dims] + other.shape`` and whose dtype follows
# ``other``, so the benchmark measures the cost of that zero allocation plus the
# feature-meta bookkeeping. Every case allocates a sizable output so the timing
# reflects actual device allocation rather than pure dispatch overhead.
_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES = [
    ((4,), (1024, 1024), 1),
    ((16,), (256, 256), 1),
    ((64,), (512, 512), 1),
    ((32,), (1024, 64), 1),
    ((4, 16), (1024, 64), 2),
    ((8, 32), (512, 128), 2),
    ((2,), (20, 320, 15), 1),
    ((8, 16), (256, 256), 2),
]


def _case_fn(shape, dtype):
    del dtype
    self_shape, other_shape, self_num_batch_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"self": self_shape, "other": other_shape},
        params={"self_num_batch_dims": self_num_batch_dims},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    self_shape, other_shape, self_num_batch_dims = plan.builder_args[0]
    self_inp = utils.generate_tensor_input(self_shape, dtype, device)
    other_inp = utils.generate_tensor_input(other_shape, dtype, device)
    return self_inp, other_inp, {"self_num_batch_dims": self_num_batch_dims}


class NewZerosWithSameFeatureMetaBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to the op's allocation shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES


@pytest.mark._new_zeros_with_same_feature_meta
def test__new_zeros_with_same_feature_meta():
    bench = NewZerosWithSameFeatureMetaBenchmark(
        op_name="_new_zeros_with_same_feature_meta",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._new_zeros_with_same_feature_meta,
        gems_op=getattr(flag_gems, "_new_zeros_with_same_feature_meta", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
