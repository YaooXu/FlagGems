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

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::atleast_1d is a pure view/identity op: a 0-dim tensor is reshaped to
# (1,) (a view of the same storage) while tensors with one or more dimensions
# are returned unchanged. No arithmetic is performed, so the result must match
# bit-for-bit, alias the input, and every storage dtype the op supports is
# covered. The value-range framework replaces plain randn input generation:
# values pass through untouched, and the shared ranges cover negative/positive/
# boundary magnitudes per dtype (int/bool are exact; floats use equal_nan).
#
# The .default overload is resolved through its public name "atleast_1d" (a
# KernelGen override_gems_op("atleast_1d", ...) wins over the direct callable);
# the .Sequence overload shares the same public name.

_FLOAT_DTYPES = list(utils.ALL_FLOAT_DTYPES)
_INT_DTYPES = [torch.int8, torch.uint8, *utils.ALL_INT_DTYPES]
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_DTYPE_CANDIDATES = _FLOAT_DTYPES + _INT_DTYPES + _FP8_DTYPES + list(utils.BOOL_TYPES)

# The op supports every storage dtype (it is a pure view), so probe the device
# once and drop any dtype the active backend cannot build/call. The fallback
# keeps parametrization non-empty so collection never fails on an empty set.
_SUPPORTED_DTYPES = (
    tu.supported_dtypes("atleast_1d", candidates=_DTYPE_CANDIDATES) or _DTYPE_CANDIDATES
)

# nan/inf pass through a view untouched; float8 is included but compared through
# the upcast path of _assert_result_equal below.
_NAN_INF_DTYPES = [
    dtype for dtype in _SUPPORTED_DTYPES if dtype in _FLOAT_DTYPES + _FP8_DTYPES
]

# The shared shape levels cover the dim boundary that drives the op: 0-dim ->
# (1,) view and 1-dim/higher identity. The 0-dim scalar is prepended defensively
# in case a level ever drops it.
_ATLEAST_1D_SHAPES = tuple(tu.selected_shapes())
if () not in _ATLEAST_1D_SHAPES:
    _ATLEAST_1D_SHAPES = ((),) + _ATLEAST_1D_SHAPES

# Backward shapes stay small (the autograd graph is built on the reference and
# the comparison is elementwise); 0-dim exercises the shape-changing view.
_ATLEAST_1D_BACKWARD_SHAPES = [(), (3,), (16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolution order: (1) the process-local override injected by KernelGen,
    # (2) the direct flag_gems.atleast_1d callable, (3) None -> the test falls
    # back to the PyTorch reference so it stays runnable before a FlagGems
    # implementation is registered. Both the .default and .Sequence overloads
    # are resolved through the shared public operator name "atleast_1d".
    try:
        return flag_gems.testing.resolve_gems_op(
            "atleast_1d", getattr(flag_gems, "atleast_1d", None)
        )
    except LookupError:
        return None


def _apply_atleast_1d(inp):
    # Called inside each test function (never at import time) so that the
    # override installed by KernelGen for this run is the one that is used. A
    # Python list dispatches to the .Sequence overload on the reference packet.
    gems_op = _resolve_gems_op()
    if gems_op is None:
        return torch.ops.aten.atleast_1d(inp)
    return gems_op(inp)


def _make_input(dtype, shape, value_range):
    if dtype == torch.uint8:
        # The shared framework resolves the negative range bound to -1, which
        # uint8 cannot represent: torch.testing.make_tensor then receives an
        # empty (clamped) interval and raises. Snap the bounds to the
        # representable interval; a degenerate range becomes a constant fill.
        low = max(0, int(tu.resolve_bound(value_range[0], dtype)))
        high = max(0, int(tu.resolve_bound(value_range[1], dtype)))
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )
    return tu.make_input(dtype, shape, value_range)


def _assert_result_equal(res_out, ref_out):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if res_out.dtype in _FP8_DTYPES:
        # torch.testing.assert_close cannot compare float8 tensors in this
        # torch build; upcast and compare the (exactly representable) values.
        torch.testing.assert_close(
            res_out.detach().cpu().to(torch.float32),
            ref_out.detach().cpu().to(torch.float32),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    else:
        tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_atleast_1d_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = _apply_atleast_1d(inp)

    # atleast_1d is a view op: the result must alias the input storage.
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_atleast_1d_nan_inf(dtype):
    # atleast_1d is a view: nan/inf/-inf/+-0.0 must pass through bit-for-bit.
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    # equal_nan=True is active on both comparison paths.
    _assert_result_equal(res_out, ref_out)


@pytest.mark.atleast_1d_sequence
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_atleast_1d_sequence(shape, value_range, dtype):
    # Mix a 0-dim scalar with the current shape so the sequence overload
    # exercises both the scalar -> (1,) view path and the identity path.
    inp = [
        _make_input(dtype, (), value_range),
        _make_input(dtype, shape, value_range),
        _make_input(dtype, shape, value_range),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_1d.Sequence(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        # atleast_1d is a view op: each result must alias its input.
        assert res.data_ptr() == src.data_ptr()
        _assert_result_equal(res, ref)


@pytest.mark.atleast_1d_sequence
def test_atleast_1d_sequence_empty():
    # A Tensor[] input may legitimately be empty: the reference returns an
    # empty list and the candidate must return an empty list too.
    ref_out = torch.ops.aten.atleast_1d.Sequence([])
    res_out = _apply_atleast_1d([])
    assert len(ref_out) == 0
    assert len(res_out) == 0


@pytest.mark.atleast_1d_backward
@pytest.mark.parametrize("shape", _ATLEAST_1D_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_1d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp)

    # atleast_1d is a view: the gradient of sum(atleast_1d(x)) is all-ones in
    # x's shape on both the shape-changing (0-dim) and identity paths.
    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out.sum(), ref_inp)[0]
    tu.assert_result_close(ref_in_grad, torch.ones_like(ref_inp))

    # The candidate forward must match the reference...
    res_out = _apply_atleast_1d(inp)
    _assert_result_equal(res_out, ref_out)

    # ...and, if the candidate view is autograd-aware (a compiled kernel that
    # returns a plain tensor is not), its gradient must match too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out.sum(), inp)[0]
        tu.assert_result_close(res_in_grad, torch.ones_like(inp))


@pytest.mark.atleast_1d_negative
def test_atleast_1d_rejects_non_tensor():
    # The aten op only accepts a Tensor (a list of Tensors goes through the
    # .Sequence overload); Python scalars hit a schema mismatch and raise.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_1d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_1d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(3.14)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op([torch.zeros(2, device=flag_gems.device), 3.14])
