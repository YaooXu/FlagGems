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

# aten::atleast_3d is a pure view/identity op: 0-dim tensors are reshaped to
# (1, 1, 1), 1-dim tensors to (1, N, 1), 2-dim tensors to (M, N, 1) (all views),
# and tensors with three or more dimensions are returned as-is. No arithmetic is
# performed, so the result must match bit-for-bit, alias the input, and every
# dtype the op supports is covered. The value-range framework replaces plain
# randn input generation: values pass through untouched, and the shared ranges
# cover negative/positive/boundary magnitudes per dtype (int/bool are exact;
# float uses equal_nan). Both the .default and .Sequence overloads are resolved
# through the shared public operator name "atleast_3d".
ATLEAST_3D_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# The shared shape levels already cover the dim boundary that drives the op:
# 0-dim -> (1, 1, 1), 1-dim -> (1, N, 1), 2-dim -> (M, N, 1), 3-dim identity
# and (in the "all" level) higher-dim identity.
_ATLEAST_3D_SHAPES = tu.selected_shapes()

# Backward shapes stay small (the autograd graph is built on the reference and
# the comparison is elementwise); 0-dim/1-dim/2-dim exercise the shape-changing
# views.
_ATLEAST_3D_BACKWARD_SHAPES = [(), (3,), (4, 5), (16, 64), (7, 13, 29)]


# Resolution order: (1) the process-local override injected by KernelGen,
# (2) the direct flag_gems.atleast_3d callable, (3) None -> the test falls
# back to the PyTorch reference so it stays runnable before a FlagGems
# implementation is registered. Both the .default and .Sequence overloads
# are resolved through the shared public operator name "atleast_3d".
def _resolve_gems_op():
    try:
        return flag_gems.testing.resolve_gems_op(
            "atleast_3d", getattr(flag_gems, "atleast_3d", None)
        )
    except LookupError:
        return None


def _apply_atleast_3d(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.atleast_3d(inp)
    return gems_op(inp)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", _ATLEAST_3D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # atleast_3d is a view op: the result must alias the input.
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_atleast_3d_nan_inf(dtype):
    # Values pass through a view untouched: nan/inf/-inf and signed zeros must
    # be preserved (assert_result_close uses equal_nan=True on the float path).
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_atleast_3d_complex(dtype):
    # atleast_3d also supports complex tensors (a pure view: real and imaginary
    # parts pass through untouched). One negative-and-positive range per dtype
    # suffices because no arithmetic is performed.
    inp = tu.make_input(dtype, (2, 5), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("shape", _ATLEAST_3D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_sequence(shape, value_range, dtype):
    # Mix a 0-dim scalar, a 1-dim tensor, a 2-dim tensor and the current shape
    # so the sequence overload exercises all four paths: scalar -> (1, 1, 1),
    # 1-dim -> (1, N, 1), 2-dim -> (M, N, 1) and the >= 3-dim identity.
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, (3,), value_range),
        tu.make_input(dtype, (4, 5), value_range),
        tu.make_input(dtype, shape, value_range),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        # atleast_3d is a view op: each result must alias its input.
        assert res.data_ptr() == src.data_ptr()
        tu.assert_result_close(res, ref)


@pytest.mark.atleast_3d_sequence
def test_atleast_3d_sequence_empty():
    # A Tensor[] input may legitimately be empty: the reference returns an
    # empty list, and the candidate must return an empty list too.
    ref_out = torch.ops.aten.atleast_3d.Sequence([])
    res_out = _apply_atleast_3d([])
    assert len(ref_out) == 0
    assert len(res_out) == 0


@pytest.mark.atleast_3d_backward
@pytest.mark.parametrize("shape", _ATLEAST_3D_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_3d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp)

    # atleast_3d is a view: the gradient of sum(atleast_3d(x)) is all-ones in
    # x's shape on both the shape-changing (0-dim/1-dim/2-dim) and identity
    # paths.
    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out.sum(), ref_inp)[0]
    tu.assert_result_close(ref_in_grad, torch.ones_like(ref_inp))

    # The candidate forward must match the reference...
    res_out = _apply_atleast_3d(inp)
    tu.assert_result_close(res_out, ref_out)

    # ...and, if the candidate view is autograd-aware (a compiled kernel that
    # returns a plain tensor is not), its gradient must match too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out.sum(), inp)[0]
        tu.assert_result_close(res_in_grad, torch.ones_like(inp))


@pytest.mark.atleast_3d_negative
def test_atleast_3d_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(3.14)
