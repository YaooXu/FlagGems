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

import flag_gems  # noqa: E402

from . import base, consts, utils  # noqa: E402

# aten::thnn_conv2d(self, weight, kernel_size, bias, stride, padding) performs
# an im2col-based 2-D convolution (groups=1, no dilation). The default shape set
# has no convolved input/weight pairs, so define local performance shapes whose
# output sizes stay in the tens-of-MB range. Each tuple is
# (inp_shape, weight_shape, kernel_size, stride, padding); the im2col cost grows
# with kernel area, so both 1x1 (pure GEMM) and 3x3/5x5 (im2col-heavy) kernels
# are represented.
THNN_CONV2D_SHAPES = [
    ((32, 64, 128, 128), (32, 64, 1, 1), (1, 1), (1, 1), (0, 0)),
    ((32, 64, 56, 56), (32, 64, 3, 3), (3, 3), (1, 1), (1, 1)),
    ((64, 32, 18, 18), (64, 32, 5, 5), (5, 5), (2, 2), (1, 1)),
    ((64, 32, 32, 32), (32, 32, 3, 3), (3, 3), (2, 2), (0, 0)),
    ((16, 128, 16, 16), (64, 128, 3, 3), (3, 3), (1, 1), (1, 1)),
]


def _case_fn(shape, dtype):
    del dtype
    inp_shape, weight_shape, kernel_size, stride, padding = shape
    yield base.BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
        },
        builder_args=(inp_shape, weight_shape, kernel_size, stride, padding),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, weight_shape, kernel_size, stride, padding = plan.builder_args
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = utils.generate_tensor_input((weight_shape[0],), dtype, device)
    return inp, weight, kernel_size, bias, stride, padding, {}


class ThnnConv2dBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding)."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = THNN_CONV2D_SHAPES


@pytest.mark.thnn_conv2d
def test_thnn_conv2d():
    bench = ThnnConv2dBenchmark(
        op_name="thnn_conv2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.thnn_conv2d,
        gems_op=getattr(flag_gems, "thnn_conv2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
