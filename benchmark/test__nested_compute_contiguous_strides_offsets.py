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

# ``_nested_compute_contiguous_strides_offsets`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so
# ``@pytest.mark._nested_compute_contiguous_strides_offsets`` and
# ``-m _nested_compute_contiguous_strides_offsets`` both work.
setattr(
    pytest.mark,
    "_nested_compute_contiguous_strides_offsets",
    MarkDecorator(
        Mark("_nested_compute_contiguous_strides_offsets", (), {}, _ispytest=True),
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

from . import base  # noqa: E402

# aten::_nested_compute_contiguous_strides_offsets(Tensor nested_size)
# -> (Tensor, Tensor) computes the contiguous strides and storage offsets of
# each sub-tensor of a nested tensor from its (num_tensors, num_dims) int64
# sizes tensor. The sizes metadata tensor is always a CPU tensor (torch creates
# it that way even for CUDA nested tensors) and the op only touches that
# metadata, so the benchmark measures dispatch plus the small int64 scan rather
# than data movement. int64 is the only dtype the op accepts (non-int64 inputs
# raise RuntimeError), so the benchmark runs int64 cases only. The default
# shape set is dominated by huge 1-D tensors that are meaningless for a
# (num_tensors, num_dims) layout, so the benchmark restricts itself to
# batch x dim layouts.
_NESTED_SIZE_SHAPES = [
    (16, 2),
    (64, 3),
    (256, 3),
    (1024, 2),
    (2048, 4),
    (4096, 5),
    (8192, 3),
]


def _case_fn(shape, dtype):
    del dtype
    num_tensors, num_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"nested_size": (num_tensors, num_dims)},
        params={"bound": 8},
        builder_args=((num_tensors, num_dims),),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    num_tensors, num_dims = plan.builder_args[0]
    gen = torch.Generator("cpu").manual_seed(0)
    nested_size = torch.randint(
        1,
        plan.params["bound"],
        (num_tensors, num_dims),
        dtype=torch.int64,
        generator=gen,
    )
    return (nested_size,)


class NestedComputeContiguousStridesOffsetsBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to nested-size layout shapes."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _NESTED_SIZE_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark._nested_compute_contiguous_strides_offsets
def test__nested_compute_contiguous_strides_offsets():
    bench = NestedComputeContiguousStridesOffsetsBenchmark(
        op_name="_nested_compute_contiguous_strides_offsets",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nested_compute_contiguous_strides_offsets,
        gems_op=getattr(flag_gems, "_nested_compute_contiguous_strides_offsets", None),
        dtypes=[torch.int64],
    )
    bench.run()
