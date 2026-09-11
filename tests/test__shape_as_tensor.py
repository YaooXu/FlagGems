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
from . import test_utils as tu

# ``_shape_as_tensor`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._shape_as_tensor`` and ``-m
# _shape_as_tensor`` both work.
setattr(
    pytest.mark,
    "_shape_as_tensor",
    MarkDecorator(Mark("_shape_as_tensor", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_shape_as_tensor(Tensor self) -> Tensor materializes the logical shape
# of ``self`` as a fresh 1-D int64 tensor. Only the rank and sizes are
# consulted; the values, dtype, strides and layout never influence the result,
# and a 0-D input yields an empty 1-D int64 tensor. aten always builds the
# output on the CPU regardless of the input device, so the candidate must do
# the same.
#
# Regular-operator-spec adaptation notes:
# - Broadcast: N/A -- the op takes a single ``self`` tensor.
# - Backward: N/A -- the output is a fresh int64 metadata tensor with no
#   autograd support, so there is no gradient to compare. A dedicated test
#   pins that a ``requires_grad`` input produces a non-grad output.
# - Value ranges: the input values are semantically irrelevant, so the
#   value-range grid below (tu.selected_ranges()) verifies that the
#   deterministic shape materialization is produced for every storage range.
# - nan/inf: covered by a dedicated case (non-finite storage values are
#   ignored; the int64 output compares exactly).
#
# Shape coverage follows the regular-operator-spec level selection (quick/all
# via the pytest --quick flag): tu.selected_shapes(), which includes the 0-D
# scalar (mapped to the empty 1-D output) and ranks up to 5.

# ---------------------------------------------------------------------------
# Dtype coverage -- probe, never guess
# ---------------------------------------------------------------------------
# The op ignores the input values and dtype, so every storage dtype family the
# runtime can allocate must be accepted. The spec's 9 required dtypes
# (int8/uint8/fp8/fp32/bf16/fp16/int32/int64 plus fp16) are probed with
# tu.supported_dtypes(); the wider float/int/bool/complex families are added
# where the probe reports support.
_EXTRA_INPUT_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)
_CANDIDATE_INPUT_DTYPES = list(
    dict.fromkeys(list(tu.REQUIRED_DTYPES) + _EXTRA_INPUT_DTYPES)
)
# If the probe yields nothing, keep the full candidate list rather than a
# float32-only fallback, so a failed/absent probe never silently drops the
# spec-required int8/uint8/fp8 dtypes.
_SHAPE_AS_TENSOR_INPUT_DTYPES = tu.supported_dtypes(
    "_shape_as_tensor", candidates=_CANDIDATE_INPUT_DTYPES
) or list(_CANDIDATE_INPUT_DTYPES)

# Zero-size dimensions are part of the logical shape and must be reported
# faithfully; a ``numel == 0`` fast path would silently drop them.
_EMPTY_SHAPES = [(0,), (0, 5), (3, 0, 4)]


def _make_input(dtype, shape, value_range):
    """tu.make_input with the unsigned-dtype fallback used across the suite.

    For unsigned dtypes a negative range bound is clamped to 0 by the shared
    helper's make_tensor call, which then rejects the degenerate interval
    (e.g. uint8 over ["-1", "0"] collapses to [0, 0]). Materialize the clamped
    interval locally instead so every spec range is still exercised.
    """
    try:
        return tu.make_input(dtype, shape, value_range)
    except RuntimeError:
        if dtype.is_floating_point or dtype.is_complex or dtype == torch.bool:
            raise
        info = torch.iinfo(dtype)
        low = max(int(tu.resolve_bound(value_range[0], dtype)), info.min)
        high = min(int(tu.resolve_bound(value_range[1], dtype)), info.max)
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins.
    # ``flag_gems._shape_as_tensor`` may not be registered yet, so getattr
    # supplies a safe default and resolve_gems_op falls back to the package
    # namespace before raising.
    return flag_gems.testing.resolve_gems_op(
        "_shape_as_tensor", getattr(flag_gems, "_shape_as_tensor", None)
    )


def _assert_result(res_out, ref_out, inp, shape):
    # The result is a fresh 1-D int64 CPU tensor holding the logical shape,
    # never a view/alias of the input.
    assert res_out.shape == ref_out.shape == (len(shape),)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == ref_out.device == torch.device("cpu")
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


def _transposed_view(base):
    return base.transpose(0, 1)


def _sliced_view(base):
    return base[0:3, 2:7, 1]


def _narrowed_view(base):
    return base.narrow(1, 1, 5)


# Each case builds a non-contiguous view of a (4, 8, 6) base and states the
# logical shape the query must report. A transposed, sliced or narrowed view
# changes the sizes, so a candidate that reads the base storage shape instead
# of the view's logical shape would fail these checks.
_VIEW_CASES = [
    ("transposed", _transposed_view, (8, 4, 6)),
    ("sliced", _sliced_view, (3, 5)),
    ("narrowed", _narrowed_view, (4, 5, 6)),
]


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_value_ranges(shape, value_range, dtype):
    # The result must be the deterministic shape materialization no matter what
    # values the storage holds, so every range from the regular-operator spec
    # is exercised here.
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_empty(shape, value_range, dtype):
    # Zero-size dimensions are part of the logical shape; a ``numel == 0`` fast
    # path that drops them would fail here.
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, shape)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SHAPE_AS_TENSOR_INPUT_DTYPES)
def test__shape_as_tensor_non_contiguous(view_case, value_range, dtype):
    # _shape_as_tensor must work on any tensor layout; only the logical shape
    # is consulted, never the storage.
    _, view_fn, expected = view_case
    base = _make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert inp.shape == expected
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, expected)


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__shape_as_tensor_nan_inf(dtype):
    # nan/inf are ordinary storage values for this op and must be ignored: the
    # result is still the deterministic shape tensor over the logical shape.
    inp = _make_input(dtype, (4, 8, 6), ["-1", "1"]).clone()
    inp[0, :, 0] = float("inf")
    inp[1, :, 1] = float("-inf")
    inp[2, :, 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, (4, 8, 6))


@pytest.mark._shape_as_tensor
@pytest.mark.parametrize("shape", [(), (1,), (2, 3, 5)])
def test__shape_as_tensor_ignores_autograd(shape):
    # The metadata query has no autograd support: a requires_grad input still
    # yields a fresh, non-grad int64 tensor with exactly the logical shape.
    inp = _make_input(torch.float32, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp.detach())

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert not res_out.requires_grad
    _assert_result(res_out, ref_out, inp.detach(), shape)


@pytest.mark._shape_as_tensor
def test__shape_as_tensor_rejects_non_tensor_input():
    # The schema requires a Tensor ``self``; every non-tensor argument is
    # rejected at binding time by aten, and the candidate must not silently
    # accept it either.
    for bad in (5, [1, 2, 3], "abc", 3.14):
        with pytest.raises(RuntimeError):
            torch.ops.aten._shape_as_tensor(bad)
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(bad)
