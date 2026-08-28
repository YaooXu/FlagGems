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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

# ``_choose_qparams_per_tensor`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._choose_qparams_per_tensor``
# and ``-m _choose_qparams_per_tensor`` both work.
setattr(
    pytest.mark,
    "_choose_qparams_per_tensor",
    MarkDecorator(
        Mark("_choose_qparams_per_tensor", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False)
# -> (float, int) computes a per-tensor quantization scale and zero_point over
# the whole input (min/max reduction), returning Python scalars (not tensors).
# Keep correctness shapes small; the reference reduction is O(numel).
CQPT_SHAPES = [(64,), (256,)] if utils.QUICK_MODE else [(64,), (1024,), (4096, 256)]
# The aten reference supports fp16/bf16/fp32/fp64 (and int), and each dtype
# changes the reduction precision, so cover all float dtypes.
CQPT_DTYPES = utils.ALL_FLOAT_DTYPES
# Input distributions that land on every branch of the quantization search:
# mixed (min < 0 < max), positive (min >= 0 -> zp = 0), negative (max <= 0 ->
# zp = qmax), constant (min == max != 0), and zero (min == max == 0 -> the
# reference clamps the scale to 0.1).
CQPT_KINDS = ["mixed", "positive", "negative", "constant", "zero"]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._choose_qparams_per_tensor is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._choose_qparams_per_tensor
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_choose_qparams_per_tensor",
        getattr(flag_gems, "_choose_qparams_per_tensor", None),
    )


def _make_input(shape, dtype, kind):
    if kind == "mixed":
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if kind == "positive":
        return torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.01
    if kind == "negative":
        return -(torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.01)
    if kind == "constant":
        return torch.full(shape, 1.5, dtype=dtype, device=flag_gems.device)
    if kind == "zero":
        return torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    raise ValueError(f"Unknown input kind: {kind}")


def _assert_scale_close(res_scale, ref_scale):
    # The op returns a Python float (scale), not a tensor, so the tensor-based
    # gems_assert_close helpers do not apply. Compare as fp64 scalar tensors:
    # atol/rtol = 1e-4 accommodates fp32 candidate arithmetic on the min/max
    # range (and the exact min-scale clamp constant, which differs by ~3.5e-8).
    torch.testing.assert_close(
        torch.tensor(res_scale, dtype=torch.float64),
        torch.tensor(ref_scale, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("shape", CQPT_SHAPES)
@pytest.mark.parametrize("dtype", CQPT_DTYPES)
@pytest.mark.parametrize("reduce_range", [False, True])
@pytest.mark.parametrize("kind", CQPT_KINDS)
def test__choose_qparams_per_tensor(shape, dtype, reduce_range, kind):
    inp = _make_input(shape, dtype, kind)
    ref_inp = utils.to_reference(inp)

    ref_scale, ref_zp = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_scale, res_zp = _resolve_gems_op()(inp, reduce_range)

    # Reference contract: a Python (float, int) pair, not tensors.
    assert isinstance(ref_scale, float)
    assert isinstance(ref_zp, int)
    assert isinstance(res_scale, float)
    assert isinstance(res_zp, int)
    _assert_scale_close(res_scale, ref_scale)
    # zero_point is round(-min/scale); with non-degenerate data the exact .5
    # boundary collision is negligible, so require exact equality.
    assert res_zp == ref_zp


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("shape", [(256,), (64, 16)])
@pytest.mark.parametrize("reduce_range", [False, True])
def test__choose_qparams_per_tensor_int32(shape, reduce_range):
    # The aten reference also accepts integer inputs (min/max reduction).
    inp = torch.randint(0, 255, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_scale, ref_zp = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_scale, res_zp = _resolve_gems_op()(inp, reduce_range)

    assert isinstance(res_scale, float)
    assert isinstance(res_zp, int)
    _assert_scale_close(res_scale, ref_scale)
    assert res_zp == ref_zp
