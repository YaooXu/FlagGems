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
# ``from . import base, consts`` cannot resolve this checkout's benchmark package
# through normal package discovery. Put the checkout root on sys.path so the
# ``benchmark`` package resolves to THIS checkout no matter how pytest is invoked
# (belt-and-suspenders: the correctness file already does this when it runs
# first, but this keeps the benchmark file self-contained).
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402

# ``_slow_conv2d_backward`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._slow_conv2d_backward`` and
# ``-m _slow_conv2d_backward`` both work.
setattr(
    pytest.mark,
    "_slow_conv2d_backward",
    MarkDecorator(
        Mark("_slow_conv2d_backward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride,
# padding, output_mask) -> (grad_input, grad_weight, grad_bias) is the im2col
# based "slow" conv2d backward (no dilation, groups always 1). ``self`` is
# (N, C_in, H, W), ``weight`` is (C_out, C_in, kH, kW) and ``grad_output`` is
# (N, C_out, H_out, W_out) with H_out = (H + 2*pH - kH) // sH + 1. Each tuple is
# (N, in_c, H, W, out_c, kH, kW, stride, padding); the im2col cost grows with
# kernel area, so both 1x1 (pure GEMM) and 2x2/3x3 (im2col-heavy) kernels are
# represented, with output sizes in the tens-of-MB range.
_SLOW_CONV2D_BACKWARD_SHAPES = [
    (16, 4, 8, 8, 4, 3, 3, 1, 0),
    (8, 3, 16, 16, 8, 3, 3, 1, 0),
    (32, 8, 8, 8, 32, 2, 2, 1, 1),
    (4, 16, 4, 4, 16, 1, 1, 2, 0),
    (16, 8, 32, 32, 16, 3, 3, 2, 1),
    (8, 16, 64, 64, 8, 3, 3, 1, 1),
]


class SlowConv2dBackwardBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark over (grad_output, input, weight, stride, padding)."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = _SLOW_CONV2D_BACKWARD_SHAPES


def _case_fn(shape, dtype):
    del dtype
    batch, in_c, h, w, out_c, k_h, k_w, stride, padding = shape
    in_shape = (batch, in_c, h, w)
    weight_shape = (out_c, in_c, k_h, k_w)
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1
    yield base.BenchmarkCasePlan(
        shape={
            "input": in_shape,
            "weight": weight_shape,
            "grad_output": (batch, out_c, h_out, w_out),
        },
        params={"stride": stride, "padding": padding},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, in_c, h, w, out_c, k_h, k_w, stride, padding = plan.builder_args[0]
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1
    grad_output = torch.randn((batch, out_c, h_out, w_out), dtype=dtype, device=device)
    input = torch.randn((batch, in_c, h, w), dtype=dtype, device=device)
    weight = torch.randn((out_c, in_c, k_h, k_w), dtype=dtype, device=device)
    return (
        grad_output,
        input,
        weight,
        (k_h, k_w),
        (stride, stride),
        (padding, padding),
        (True, True, True),
    )


@pytest.mark._slow_conv2d_backward
def test__slow_conv2d_backward():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = SlowConv2dBackwardBenchmark(
        op_name="_slow_conv2d_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._slow_conv2d_backward.output_mask,
        gems_op=getattr(flag_gems, "_slow_conv2d_backward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
