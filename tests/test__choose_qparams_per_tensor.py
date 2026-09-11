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

"""Accuracy tests for ``aten::_choose_qparams_per_tensor``.

``aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False)
-> (float, int)`` reduces the whole input to ``(min, max)`` and returns a
per-tensor affine-quantization ``(scale, zero_point)`` pair as Python scalars
(not tensors). The reference contract, identical on CPU and CUDA for every
dtype the op accepts, is::

    dmin, dmax = float(input.min()), float(input.max())   # stored values
    qmax       = 127 if reduce_range else 255
    raw        = (max(dmax, 0) - min(dmin, 0)) / qmax
    raw == 0.0      -> scale = 0.1,   zero_point = 0
    raw <  6.1e-5   -> scale = 6.1e-5               (clamped)
    otherwise       -> scale = raw
    zero_point = round(-min(dmin, 0) / raw)         (rounded from the
                                                     UNCLAMPED raw scale)

Value-range coverage uses the shared ``tests/test_utils.py`` framework: the
five required ranges ``[-1,1] [0,1] [-1,0] [0,max] [min,0]`` over the seven
required shapes and every dtype the reference accepts. All dtypes the op
supports are exercised: float16 / float32 / bfloat16 / float64, int8 / uint8 /
int16 / int32 / int64 and bool; fp8 and complex are rejected by the reference
(negative cases below). ``reduce_range`` is swept on every value-range case.

Three backend quirks are handled by the helpers below:

* the CUDA reference converts the min/max reduction to fp32 internally, so an
  fp64 ``finfo.max`` range (~1.8e308) makes the *reference itself* raise
  ("value cannot be converted to type float without overflow"); fp64 works
  fine below that bound, so the helper substitutes ``1e30`` for the extreme
  symbols instead of dropping fp64;
* ``uint8`` cannot go through ``tu.make_input``: a negative lower bound is
  passed straight to ``torch.testing.make_tensor``, which rejects it for
  unsigned dtypes ("random_ expects 'from' to be less than 'to'"). The helper
  clamps unsigned bounds to 0;
* both are resolved inside the tests (never at import time) so KernelGen can
  inject the candidate through ``override_gems_op``.

Besides the value-range grid the file pins the clamp / ``raw == 0`` branches
(tiny and constant inputs), the accepted inf path, non-contiguous strided
inputs, the default ``reduce_range`` argument, and the negative cases (nan,
empty, complex, fp8, non-tensor).
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``_choose_qparams_per_tensor`` starts with an underscore and ``pytest.mark``
# refuses attribute access for such names, so register the marker directly on
# the MarkGenerator: ``@pytest.mark._choose_qparams_per_tensor`` and
# ``-m _choose_qparams_per_tensor`` then both work.
setattr(
    pytest.mark,
    "_choose_qparams_per_tensor",
    MarkDecorator(
        Mark("_choose_qparams_per_tensor", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# Dtype coverage: every dtype the reference accepts (probed on the active
# device). ALL_FLOAT_DTYPES / ALL_INT_DTYPES already honour the device's
# bf16 / fp64 / int64 support flags.
_CQPT_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

_CQPT_REDUCE_RANGE = [False, True]

# Small fp32 constants landing on the clamp / rounding branches. Ratios are
# kept away from exact half-integers so any faithful (fp32- or fp64-arithmetic)
# candidate reproduces the reference zero_point exactly.
_CQPT_TINY_CASES = [
    [0.005],  # raw ~= 1.96e-5 -> clamped, zp = 0
    [-0.005],  # raw ~= 1.96e-5 -> clamped, zp = qmax
    [-0.0075, 0.0025, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 191 / 95
    [-0.002, 0.008, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 51 / 25
    [-0.008, 0.002, 0.0],  # raw ~= 3.92e-5 -> clamped, zp = 204 / 102
    [0.016],  # raw ~= 6.27e-5 -> just above the clamp, unclamped
]

# Constant inputs pin the degenerate min == max branches: all-zero -> scale
# 0.1 / zp 0, constant positive -> zp 0, constant negative -> zp qmax.
_CQPT_CONSTANT_VALUES = [0.0, 5.0, -5.0, 1e-8, 1e-2]

# inf/-inf are accepted by the reference: scale = inf, zp = INT32_MIN (the
# fp32-to-int32 cast of the nan produced by inf/inf).
_CQPT_INF_INPUT = [float("inf"), float("-inf"), 0.0]

# fp64 extreme symbols are bounded below the fp32-internal reference overflow.
_FP64_EXTREME_BOUND = 1e30


def _fp8_available():
    """True when the active device can materialise an fp8 tensor at all."""
    try:
        torch.zeros((1,), dtype=torch.float8_e4m3fn, device=flag_gems.device)
    except Exception:
        return False
    return True


_FP8_MARK = pytest.mark.skipif(
    not _fp8_available(), reason="fp8 tensors are not supported on this device"
)


def _make_input(shape, dtype, value_range):
    """tu.make_input with the uint8 / fp64 quirks described in the docstring."""
    if dtype == torch.uint8:
        # Clamp negative symbols to 0 for unsigned dtypes.
        low = max(int(tu.resolve_bound(value_range[0], dtype)), 0)
        high = max(int(tu.resolve_bound(value_range[1], dtype)), 0)
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )

    if dtype != torch.float64:
        return tu.make_input(dtype, shape, value_range)

    # fp64: keep |value| <= 1e30 so the fp32-internal reference does not
    # overflow on the finfo-derived "min" / "max" symbols.
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
    """Resolve the candidate inside the test (never at import time).

    Resolution order: (1) the process-local KernelGen override,
    (2) the direct ``flag_gems._choose_qparams_per_tensor`` callable,
    (3) LookupError. The op has no native FlagGems kernel yet, so the candidate
    is always injected by KernelGen through ``override_gems_op``.
    """
    return flag_gems.testing.resolve_gems_op(
        "_choose_qparams_per_tensor",
        getattr(flag_gems, "_choose_qparams_per_tensor", None),
    )


def _assert_pair(res, ref):
    """Compare a candidate ``(scale, zero_point)`` pair against the reference.

    Both pairs must be Python ``(float, int)`` tuples. ``scale`` is compared as
    fp64 with the tolerance the op's fp32-internal reference arithmetic needs;
    ``zero_point`` must match exactly (the razor-thin exact half-integer
    ratios are deliberately not pinned by any case in this file).
    """
    res_scale, res_zp = res
    ref_scale, ref_zp = ref
    assert isinstance(ref_scale, float), type(ref_scale)
    assert isinstance(ref_zp, int), type(ref_zp)
    assert isinstance(res_scale, float), type(res_scale)
    assert isinstance(res_zp, int), type(res_zp)
    torch.testing.assert_close(
        torch.tensor(res_scale, dtype=torch.float64),
        torch.tensor(ref_scale, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )
    assert res_zp == ref_zp, f"zero_point {res_zp} != {ref_zp}"


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CQPT_DTYPES)
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test__choose_qparams_per_tensor_value_ranges(
    shape, value_range, dtype, reduce_range
):
    """The required 5 ranges x 7 shapes x supported-dtypes x reduce_range grid."""
    utils.init_seed(0)
    inp = _make_input(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = _resolve_gems_op()(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("values", _CQPT_TINY_CASES)
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test__choose_qparams_per_tensor_tiny_scale(values, reduce_range):
    """Pin the min-scale clamp, the raw == 0 -> 0.1 branch and the fact that
    zero_point rounds from the UNCLAMPED raw scale just below the clamp
    boundary."""
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = _resolve_gems_op()(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("value", _CQPT_CONSTANT_VALUES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test__choose_qparams_per_tensor_constant(value, dtype, reduce_range):
    """min == max branches: exact zero -> (0.1, 0); constant positive -> zp 0;
    constant negative -> zp qmax; tiny constant -> clamped scale."""
    inp = torch.full((1024,), value, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = _resolve_gems_op()(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test__choose_qparams_per_tensor_inf(reduce_range):
    """inf / -inf are accepted: scale = inf, zero_point = INT32_MIN."""
    inp = torch.tensor(_CQPT_INF_INPUT, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = _resolve_gems_op()(inp, reduce_range)

    assert ref_pair[0] == float("inf")
    assert ref_pair[1] == torch.iinfo(torch.int32).min
    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("layout", ["transpose", "slice"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduce_range", _CQPT_REDUCE_RANGE)
def test__choose_qparams_per_tensor_non_contiguous(layout, dtype, reduce_range):
    """The min/max reduction must honour arbitrary strides."""
    utils.init_seed(0)
    base_inp = torch.randn((64, 32), dtype=dtype, device=flag_gems.device)
    inp = base_inp.t() if layout == "transpose" else base_inp[:, ::2]
    assert not inp.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp, reduce_range)
    res_pair = _resolve_gems_op()(inp, reduce_range)

    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test__choose_qparams_per_tensor_default_reduce_range(dtype):
    """The second argument is optional and defaults to False."""
    utils.init_seed(0)
    inp = _make_input((20, 320, 15), dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_pair = torch.ops.aten._choose_qparams_per_tensor(ref_inp)
    res_pair = _resolve_gems_op()(inp)

    _assert_pair(res_pair, ref_pair)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_nan():
    """The reference validates min <= max and raises on nan; the candidate must
    fail too rather than silently emit a nan scale."""
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
    """The reduction of a 0-element tensor is undefined; both paths must reject
    it."""
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_complex():
    """min/max reduction is not implemented for complex; reject the dtype."""
    inp = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
@_FP8_MARK
def test__choose_qparams_per_tensor_rejects_fp8():
    """fp8 has no min/max reduction kernel; reject the dtype."""
    inp = torch.zeros((4,), dtype=torch.float8_e4m3fn, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._choose_qparams_per_tensor(ref_inp, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, False)


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor_rejects_non_tensor():
    """The aten op requires a Tensor; the candidate must fail too rather than
    silently accept scalars."""
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._choose_qparams_per_tensor(3.14, False)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(3.14, False)
