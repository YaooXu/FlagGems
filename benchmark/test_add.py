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

# KernelGen's in-process verification stages the benchmark files into an
# isolated temp copy of the checkout, where the relative ``from . import base``
# cannot resolve this checkout's benchmark package through normal package
# discovery. Put the checkout root on sys.path so the ``benchmark`` package
# resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# aten::add is a binary pointwise op, so BinaryPointwiseBenchmark covers its
# timing semantics. The default consts.DEFAULT_SHAPES contains a 2**30-element
# 1-dim shape (4 GiB per fp32 tensor), which OOMs the GPU during materialization;
# this subclass pins bounded, performance-relevant shapes instead (square/wide
# 2-dim, 3-dim, 4-dim and the canonical 20x320x15 attention shape).
_ADD_BENCH_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (1024, 4096),
    (64, 512, 512),
    (16, 128, 64, 60),
    (20, 320, 15),
]


class _AddBenchmark(base.BinaryPointwiseBenchmark):
    def set_more_shapes(self):
        # No additional comprehensive-level shapes: the pinned set above is the
        # complete coverage for this benchmark.
        return []

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [tuple(s) for s in _ADD_BENCH_SHAPES]


@pytest.mark.add
def test_add():
    bench = _AddBenchmark(
        op_name="add",
        torch_op=torch.ops.aten.add,
        gems_op=flag_gems.add,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.add_
def test_add_inplace():
    bench = _AddBenchmark(
        op_name="add_",
        torch_op=torch.ops.aten.add_,
        gems_op=flag_gems.add_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
