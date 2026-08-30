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
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package (and, for the sibling benchmark file, ``benchmark``) resolve
# to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

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
# -> (float, int) computes a per-tensor quantization (scale, zero_point) pair
# over a whole-input min/max reduction, returning Python scalars (not tensors).
# The reference contract (identical on CPU and CUDA) is:
#   dmin/dmax  = input.min() / input.max()          (exact stored values)
#   qmax       = 127 if reduce_range else 255
#   raw        = (max(dmax, 0) - min(dmin, 0)) / qmax
#   raw == 0.0 -> scale = 0.1, zero_point = 0       (all-zero input)
#   raw <  6.1e-5 -> scale = 6.1e-5 (clamped)       (min normal fp16)
#   otherwise  -> scale = raw
#   zero_point = round(-min(dmin, 0) / raw)         (integer rounding of the
#                                                    UNCLAMPED raw scale, not
#                                                    the clamped one)
# The clamp constant is 6.0999998822808266e-05. On every non-degenerate input
# the arithmetic is exactly reproducible from dmin/dmax in fp64; the only
# inputs where a faithful fp64 implementation can differ from the reference by
# 1 in zero_point are exact half-integer ratios (b/raw == k + 0.5), where even
# two fp64 division roundings can land on different sides of the .5 boundary
# (empirically, e.g. [-0.005, 0.005, 0.0] with reduce_range=False gives 127
# while the exact ratio is 127.5 -> 128 under round-half-even). Those razor
# cases are deliberately NOT pinned here: the clamp and the raw-scale rounding
# are covered by the non-degenerate cases below, which are reproducible in any
# precision model. nan raises RuntimeError ("In ChooseQuantizationParams, min
# should be less than or equal to max"), inf/-inf are accepted (scale = inf,
# zero_point = INT32_MIN), and the .default overload is resolved through its
# public name "_choose_qparams_per_tensor" (KernelGen's
# override_gems_op("_choose_qparams_per_tensor", ...) wins over the direct
# callable).
#
# Dtype coverage: the op is defined for float (fp16/fp32/bf16/fp64), int
# (int16/int32/int64) and bool inputs (min/max reduction); the value-range tests
# run over all of them. The value ranges below map onto every branch of the
# quantization search: mixed (min < 0 < max), positive (min >= 0 -> zp = 0),
# negative (max <= 0 -> zp = qmax), constant (min == max != 0) and zero
# (min == max == 0 -> scale 0.1). The dedicated tiny-scale test pins the clamp
# and the raw-scale zero_point quirk; the inf test pins the accepted inf path;
# the empty/nan/complex/non-tensor tests are the negative cases.
_CQPT_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Small fp32 constants (all far below a scale of 1) that land on the clamp and
# rounding branches of the reference formula above. Ratios are kept away from
# exact half-integers so any faithful (fp32- or fp64-arithmetic) candidate
# reproduces the reference zero_point exactly.
_CQPT_TINY_CASES = [
    [0.005],  # raw ~= 1.96e-5  -> clamped scale, zp = 0
    [-0.005],  # raw ~= 1.96e-5  -> clamped scale, zp = qmax
    [-0.0075, 0.0025, 0.0],  # raw ~= 3.92e-5  -> clamped, zp = 191 / 95
    [-0.002, 0.008, 0.0],  # raw ~= 3.92e-5  -> clamped, zp = 51 / 25
    [-0.008, 0.002, 0.0],  # raw ~= 3.92e-5  -> clamped, zp = 204 / 102
    [0.016],  # raw ~= 6.27e-5  -> just above clamp, unclamped
]

# inf/-inf are accepted by the reference: scale = inf, zp = INT32_MIN.
_CQPT_INF_INPUT = [float("inf"), float("-inf"), 0.0]

# The reference's CUDA kernel converts the min/max reduction to fp32 internally,
# so a full fp64 finfo-extreme range (make_input resolves "max" to ~1.8e308)
# makes the REFERENCE itself raise ("value cannot be converted to type float
# without overflow" for |v| > ~3.4e38). fp64 is still fully supported below
# that bound (probed: 1e30 works), so the value-ranges test bounds fp64
# extremes to this magnitude instead of dropping fp64 from the cross product.
_FP64_EXTREME_BOUND = 1e30


def _make_input(shape, dtype, value_range):
    """tu.make_input, except fp64 extreme symbols resolve to _FP64_EXTREME_BOUND
    so the fp64 branches of the reference stay inside its fp32-internal range."""
    if dtype != torch.float64:
        return tu.make_input(dtype, shape, value_range)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": _FP64_EXTREME_BOUND,
        "min": -_FP64_EXTREME_BOUND,
        "max/2": _FP64_EXTREME_BOUND / 2,
        "min/2": -_FP64_EXTREME_BOUND / 2,
    }
    low = table[value_range[0]]
    high = table[value_range[1]]
    if low == high:
        return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems._choose_qparams_per_tensor
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_choose_qparams_per_tensor",
        getattr(flag_gems, "_choose_qparams_per_tensor", None),
    )


def _assert_scale_close(res_scale, ref_scale):
    # The op returns a Python float (scale), not a tensor, so the tensor-based
    # gems_assert_close helpers do not apply. Compare as fp64 scalar tensors:
    # atol/rtol = 1e-4 accommodates fp32 candidate arithmetic on the min/max
    # range (and the exact min-scale clamp constant, which differs by ~1e-8).
    torch.testing.assert_close(
        torch.tensor(res_scale, dtype=torch.float64),
        torch.tensor(ref_scale, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CQPT_DTYPES)
@pytest.mark.parametrize("reduce_range", [False, True])
def test__choose_qparams_per_tensor_value_ranges(
    shape, value_range, dtype, reduce_range
):
    inp = _make_input(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_scale, ref_zp = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_scale, res_zp = _resolve_gems_op()(inp, reduce_range)

    # Reference contract: a Python (float, int) pair, not tensors.
    assert isinstance(ref_scale, float)
    assert isinstance(ref_zp, int)
    assert isinstance(res_scale, float)
    assert isinstance(res_zp, int)
    _assert_scale_close(res_scale, ref_scale)
    # zero_point is round(-min/raw); with non-degenerate data the exact .5
    # boundary collision is negligible, so require exact equality.
    assert res_zp == ref_zp


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("values", _CQPT_TINY_CASES)
@pytest.mark.parametrize("reduce_range", [False, True])
def test__choose_qparams_per_tensor_tiny_scale(values, reduce_range):
    # Pins the min-scale clamp (6.1e-5), the raw == 0 -> 0.1 special case and
    # the fact that zero_point rounds from the UNCLAMPED raw scale right below
    # the clamp boundary. fp32 is the dtype the quantization scheme targets.
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_scale, ref_zp = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_scale, res_zp = _resolve_gems_op()(inp, reduce_range)

    assert isinstance(res_scale, float)
    assert isinstance(res_zp, int)
    _assert_scale_close(res_scale, ref_scale)
    assert res_zp == ref_zp


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("reduce_range", [False, True])
def test__choose_qparams_per_tensor_inf(reduce_range):
    # inf/-inf/0 is accepted: scale = inf, zero_point = INT32_MIN (the fp32-to-
    # int32 cast of the nan computed from inf/inf, as produced by the reference).
    inp = torch.tensor(_CQPT_INF_INPUT, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_scale, ref_zp = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_scale, res_zp = _resolve_gems_op()(inp, reduce_range)

    assert ref_scale == float("inf")
    assert ref_zp == torch.iinfo(torch.int32).min
    assert isinstance(res_scale, float)
    assert isinstance(res_zp, int)
    _assert_scale_close(res_scale, ref_scale)
    assert res_zp == ref_zp


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_nan():
    # The reference validates min <= max and raises on nan; the candidate must
    # fail too rather than silently emit a nan scale.
    inp = torch.tensor(
        [float("nan"), 1.0, 2.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_empty():
    # The reduction of a 0-element tensor is undefined; both the reference and
    # the candidate must reject it.
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_complex():
    # min/max reduction is not implemented for complex; reject the dtype.
    inp = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(3.14, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(3.14, False)
