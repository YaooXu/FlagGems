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

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::can_cast(ScalarType from_, ScalarType to) -> bool answers whether a value
# of ScalarType ``from_`` can be cast to ScalarType ``to`` according to aten's
# cast-safety rules. It is a pure dtype-metadata query: no tensor is created,
# the device is never touched, and the result is a plain Python bool.
#
# Regular-operator spec dimension applicability:
#   * value ranges -- N/A: the op takes two ScalarType arguments and never a
#     tensor, so there is no payload to sweep (tu.make_input / selected_ranges
#     do not apply). The dtype cross product below is the analogue: it covers
#     every family pair (bool/integral/floating/complex) in both directions,
#     including the diagonal (every type can cast to itself) and the asymmetric
#     cases such as float -> int.
#   * shape levels -- N/A: no tensor shapes exist for the op.
#   * broadcast    -- N/A: the op is a unary-dtype function returning a scalar.
#   * backward     -- N/A: the op is not differentiable and builds no graph.
#   * negative     -- covered: non-ScalarType arguments (str/None/float/list)
#     raise on the reference and must raise on the candidate too.
#   * nan/inf      -- N/A: no tensor payload.
_CAN_CAST_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]
if hasattr(torch, "complex32"):
    _CAN_CAST_DTYPES.append(torch.complex32)
_CAN_CAST_DTYPES += [torch.complex64, torch.complex128]


def _can_cast_dtypes():
    # Level-aware dtype coverage: quick keeps one representative ScalarType per
    # family (bool/integral/floating/complex); core/all cover every standard
    # ScalarType the runtime exposes so the full cross product runs.
    if tu.LEVEL == "quick":
        return [torch.bool, torch.int32, torch.float16, torch.float32, torch.complex64]
    return _CAN_CAST_DTYPES


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.can_cast is registered; resolution order is: (1) override,
    # (2) the direct flag_gems.can_cast callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "can_cast", getattr(flag_gems, "can_cast", None)
    )


def _resolve_gems_op_or_none():
    """Like _resolve_gems_op, but None while no candidate is registered yet."""
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _as_bool(value):
    # The reference returns a plain Python bool; a candidate may equivalently
    # return a 0-dim bool tensor. Normalize both before comparing.
    if isinstance(value, torch.Tensor):
        assert value.numel() == 1
        return bool(value.item())
    return bool(value)


def _assert_result(res_out, ref_out):
    # can_cast returns a plain Python bool, so exact equality is required and no
    # tolerance is involved.
    assert isinstance(ref_out, bool)
    assert isinstance(res_out, (bool, torch.Tensor))
    res_bool = _as_bool(res_out)
    ref_bool = _as_bool(ref_out)
    assert res_bool == ref_bool
    utils.gems_assert_equal(torch.tensor(res_bool), torch.tensor(ref_bool))


@pytest.mark.can_cast
@pytest.mark.parametrize("from_dtype", _can_cast_dtypes())
@pytest.mark.parametrize("to_dtype", _can_cast_dtypes())
def test_can_cast(from_dtype, to_dtype):
    # Cross product over every standard ScalarType in both directions. Each
    # (from_, to) pair is one workload; the expected outcome comes from the
    # reference and the candidate must agree on both True (same-family
    # widening, bool -> float, complex widening) and False (float -> int,
    # int -> float16, complex -> float) cases.
    ref_out = torch.ops.aten.can_cast(from_dtype, to_dtype)
    res_out = _resolve_gems_op()(from_dtype, to_dtype)

    _assert_result(res_out, ref_out)


# Non-ScalarType arguments: the aten schema requires ScalarType (an int at the
# dispatcher level) for both arguments; str/None/float/list hit the invalid
# argument-combination path and raise RuntimeError on the reference.
_INVALID_SCALARTYPE_CASES = [
    pytest.param("float32", id="str"),
    pytest.param(None, id="none"),
    pytest.param(3.14, id="float"),
    pytest.param([1, 2], id="list"),
]


@pytest.mark.can_cast
@pytest.mark.parametrize("bad_arg", _INVALID_SCALARTYPE_CASES)
def test_can_cast_rejects_non_scalartype_from(bad_arg):
    # A candidate must fail loudly on a non-ScalarType ``from_`` instead of
    # silently returning a bogus bool.
    with pytest.raises(RuntimeError):
        torch.ops.aten.can_cast(bad_arg, torch.float32)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        # A plain-Python candidate naturally raises TypeError/ValueError (or an
        # AttributeError) for the same inputs, which is equally acceptable.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(bad_arg, torch.float32)


@pytest.mark.can_cast
@pytest.mark.parametrize("bad_arg", _INVALID_SCALARTYPE_CASES)
def test_can_cast_rejects_non_scalartype_to(bad_arg):
    # Same contract for the ``to`` argument.
    with pytest.raises(RuntimeError):
        torch.ops.aten.can_cast(torch.float32, bad_arg)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(torch.float32, bad_arg)
