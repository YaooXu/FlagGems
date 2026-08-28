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

# aten::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding,
# output_padding, dilation) performs an im2col-based 2-D transposed convolution
# (groups=1) with output_padding/dilation support. The default shape set has no
# transposed-conv input/weight pairs, so define local performance shapes whose
# output sizes stay in the tens-of-MB range. Each tuple is (inp_shape,
# weight_shape, kernel_size, stride, padding, output_padding, dilation); note
# the transposed weight layout (C_in, C_out, kH, kW). 1x1 (pure GEMM), 3x3/5x5
# (im2col-heavy), stride-2, output_padding, and dilation-2 cases are all
# represented.
SLOW_CONV_TRANSPOSE2D_SHAPES = [
    ((32, 64, 128, 128), (64, 64, 3, 3), (3, 3), (1, 1), (0, 0), (0, 0), (1, 1)),
    ((32, 64, 56, 56), (64, 64, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1), (1, 1)),
    ((16, 64, 56, 56), (64, 128, 3, 3), (3, 3), (1, 1), (1, 1), (0, 0), (2, 2)),
    ((8, 128, 32, 32), (128, 128, 3, 3), (3, 3), (1, 1), (1, 1), (0, 0), (1, 1)),
    ((8, 64, 32, 32), (64, 128, 5, 5), (5, 5), (2, 2), (2, 2), (1, 1), (1, 1)),
    ((16, 32, 64, 64), (32, 64, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1), (2, 2)),
    ((16, 64, 56, 56), (64, 64, 1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (1, 1)),
    ((8, 64, 64, 64), (64, 64, 3, 3), (3, 3), (2, 2), (2, 2), (1, 1), (1, 1)),
    ((16, 32, 112, 112), (32, 64, 3, 3), (3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
]


def _case_fn(shape, dtype):
    del dtype
    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation = (
        shape
    )
    yield base.BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
            "output_padding": output_padding,
            "dilation": dilation,
        },
        builder_args=(
            inp_shape,
            weight_shape,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation = (
        plan.builder_args
    )
    inp = utils.generate_tensor_input(inp_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = utils.generate_tensor_input((weight_shape[1],), dtype, device)
    return (
        inp,
        weight,
        kernel_size,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        {},
    )


class SlowConvTranspose2dBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark over (input, weight, kernel, stride, padding, output_padding, dilation)."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOW_CONV_TRANSPOSE2D_SHAPES


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d():
    bench = SlowConvTranspose2dBenchmark(
        op_name="slow_conv_transpose2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.slow_conv_transpose2d,
        gems_op=getattr(flag_gems, "slow_conv_transpose2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
