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

import pytest
import torch

import flag_gems

# The KernelGen harness runs pytest in-process with its own ``tests`` package
# (kernelgen/tests) earlier on sys.path than this checkout's ``tests`` package.
# With ``--import-mode=importlib`` pytest does not prepend the checkout root, so
# ``tests`` would resolve to the harness's package and ``from . import
# accuracy_utils`` would fail with ImportError during collection. Re-point the
# ``tests`` package at this file's directory before importing the helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests as _tests_pkg  # noqa: E402

if _HERE not in getattr(_tests_pkg, "__path__", []):
    sys.modules.pop("tests", None)
    import tests as _tests_pkg  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::diagflat(Tensor self, int offset=0) -> Tensor flattens ``self`` into a
# 1-D vector (in logical row-major view order) and returns a NEW 2-D square
# matrix whose ``offset``-th diagonal holds that vector while every other entry
# is zero. The output side length is numel(self) + |offset|, so the output is
# quadratic in the input element count: correctness shapes are therefore bounded
# to keep the output under ~1M elements, and offsets whose magnitude may exceed
# numel are covered by a dedicated test. The op never transforms the stored
# values, so every storage dtype aten supports is exercised (float incl. float64
# when available, int, and bool) and nan/inf/+-0.0 round-trip unchanged.
#
# Coverage follows the regular-operator spec adapted to a pure data-movement op:
#   * shape levels: bounded shapes (0-D up to 5-D plus the empty input) merged
#     with the small shapes from tu.selected_shapes(); the generic multi-dim
#     levels whose numel would make the quadratic output explode are excluded;
#   * value ranges: tu.selected_ranges() over small representative shapes for
#     every supported dtype (the values must round-trip exactly through the
#     diagonal placement);
#   * edge cases: empty inputs, large offsets (|offset| > numel), non-contiguous
#     (transposed and strided) inputs, and nan/inf/-inf/+-0.0 passthrough;
#   * backward: autograd.grad() against the analytic diag(grad_out, offset)
#     gradient (a diagonal extraction with no arithmetic);
#   * negative: non-tensor inputs and non-int offsets raise on both paths.
_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_INT_DTYPES = utils.ALL_INT_DTYPES
_DIAGFLAT_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + utils.BOOL_TYPES

_DIAGFLAT_OFFSETS = [-2, -1, 0, 1, 2]

# Bounded shape levels: (32, 32) is the largest input (numel 1024 -> output
# side 1026, ~1M output elements, at the correctness cap).
_DIAGFLAT_SHAPES = [
    (),
    (1,),
    (16,),
    (257,),
    (0,),
    (2, 3),
    (32, 32),
    (4, 5, 6),
    (2, 3, 4, 5),
    (2, 2, 2, 2, 3),
]

# Small inputs for the value-range sweep: diagflat is a pure diagonal gather, so
# a few sizes suffice to exercise the full spec range list per dtype.
_DIAGFLAT_RANGE_SHAPES = [(8,), (2, 3), (4, 5, 6)]

_DIAGFLAT_NONCONTIG_SHAPES = [(4, 8), (6, 3), (2, 3, 4)]

_DIAGFLAT_STRIDED_SHAPES = [(16, 32), (4, 8, 16)]

_DIAGFLAT_BACKWARD_SHAPES = [(8,), (2, 3), (4, 5, 6)]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _diagflat_shapes():
    """Bounded shape levels for the main sweep.

    The generic multi-dim levels from ``tu.selected_shapes()`` (e.g.
    ``(1024, 1024)`` or ``(7, 13, 29)``) would make the quadratic diagflat
    output explode, so only the shapes whose numel keeps the output under ~1M
    elements are merged into the dedicated bounded set above.
    """
    if tu.LEVEL == "quick":
        base = [(2, 19, 7)]
    else:
        base = list(_DIAGFLAT_SHAPES)
    for shape in tu.selected_shapes():
        if _numel(shape) <= 1024 and shape not in base:
            base.append(shape)
    return base


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.diagflat callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "diagflat", getattr(flag_gems, "diagflat", None)
    )


def _assert_output(res_out, ref_out, dtype):
    # diagflat materializes a new contiguous tensor (never an aliasing view):
    # the shape, dtype, contiguity, view-ness and the diagonal placement must
    # all match the aten reference.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _diagflat_shapes())
@pytest.mark.parametrize("offset", _DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    # Shape levels x offsets x every supported dtype, with values drawn from the
    # default [-1, 1] range (negative and positive for each dtype). 0-D, 1-D,
    # empty, 2-D, 3-D, 4-D and 5-D inputs are all covered.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_value_ranges(shape, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate constant ranges) must round-trip
    # exactly through the diagonal placement. bool ignores the range and is
    # covered by the shape-level test above.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(2,), (16,)])
@pytest.mark.parametrize("offset", [-7, -3, 3, 7])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_large_offset(shape, offset, dtype):
    # Offsets whose magnitude may exceed the number of elements: the flattened
    # vector must be placed on a diagonal that starts past the main diagonal,
    # leaving extra zero rows/columns around it.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_NONCONTIG_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_non_contiguous(shape, offset, dtype):
    # diagflat flattens the logical view, so a transposed (non-contiguous)
    # input must produce a different diagonal order than a contiguous one.
    # Transpose on both the test device and the reference device so the two
    # inputs share the same memory layout.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    inp = inp.transpose(-1, -2)
    ref_inp = ref_inp.transpose(-1, -2)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_STRIDED_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_strided(shape, offset, dtype):
    # A strided slice (non-unit strides along the last dim) must be flattened
    # in logical view order too, so the candidate must read through the input's
    # actual strides. Slice on both devices so the layouts match.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_diagflat_nan_inf(dtype):
    # diagflat is a pure data-movement op: +inf/-inf/nan/+-0.0 pass through
    # unchanged onto the diagonal (equal_nan=True is active on the float path of
    # assert_result_close; 1e30 overflows to inf in fp16/bf16 on both paths
    # identically).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.diagflat(ref_inp, 1)
    res_out = _resolve_gems_op()(values, 1)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [-4, -1, 0, 1, 4])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_empty_input(offset, dtype):
    # An empty input has no elements to place: offset 0 yields a 0x0 output and
    # |offset| > 0 yields an all-zero |offset| x |offset| matrix.
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_BACKWARD_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_diagflat_backward(shape, offset, dtype):
    # The forward op places flat_inp[k] at out[k, k+offset], so
    # d(diagflat(x))/dx extracts the offset-th diagonal of the grad_output and
    # reshapes it back to the input shape (a pure gather, no arithmetic).
    # Validate the autograd reference against that analytic value, then check
    # the candidate forward output and - only when the candidate output is
    # differentiable - its gradient against the reference gradient.
    n = _numel(shape)
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (n + abs(offset), n + abs(offset)), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    if dtype in (torch.float32, torch.float64):
        expected = torch.ops.aten.diag(ref_grad, offset).reshape(shape)
        tu.assert_result_close(ref_in_grad, expected)

    res_out = _resolve_gems_op()(inp, offset)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.diagflat
def test_diagflat_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(3.14)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(3.14)


@pytest.mark.diagflat
def test_diagflat_rejects_non_int_offset():
    # The schema demands an int offset; passing a float must raise on both
    # paths.
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(ref_inp, 1.5)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, RuntimeError)):
        gems_op(inp, 1.5)
