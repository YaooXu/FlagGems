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

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import base, consts, utils`` cannot resolve this checkout's benchmark
# package through normal package discovery. Put the checkout root on sys.path so
# the ``benchmark`` package resolves to THIS checkout no matter how pytest is
# invoked (belt-and-suspenders: the correctness file already does this when it
# runs first, but this keeps the benchmark file self-contained).
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts, utils  # noqa: E402

# ``_new_zeros_with_same_feature_meta`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so
# ``@pytest.mark._new_zeros_with_same_feature_meta`` and ``-m
# _new_zeros_with_same_feature_meta`` both work.
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_new_zeros_with_same_feature_meta(self, other, self_num_batch_dims=N)
# allocates a zero tensor whose shape is ``self.shape[:N] + other.shape``: the
# first N dims are taken from ``self`` (the batch prefix) and the rest from
# ``other`` (the feature dims), so self stays small while other carries the
# large feature shape. Each tuple below is a (self_shape, other_shape, N) case
# whose output size stays in the tens of MB range. The default shape set is not
# used: it contains a 1-B-element 1-D tensor whose allocation would dominate
# timing, and the op's own cost is dominated by the zero fill. self_num_batch_dims
# is keyword-only, so it is passed through the kwargs dict returned by
# build_inputs_fn.
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
    """Two-phase GenericBenchmark over (self_shape, other_shape, N) cases."""

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
