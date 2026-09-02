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
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::dstack(Tensor[] tensors) -> Tensor views every input as 3-D (atleast_3d:
# 0-dim -> (1, 1, 1), 1-dim -> (1, N, 1), 2-dim -> (M, N, 1), ndim >= 3 kept
# as-is) and concatenates the resulting tensors along the new depth axis
# (dim 2). All dims except dim 2 must match; the depth dim may vary per input.
# It is a pure data-movement op (no arithmetic), so the values round-trip
# bit-for-bit and every storage dtype aten supports is covered (float incl.
# float64 when available, int, bool and complex), with nan/inf/-inf/+-0.0
# passing through unchanged.
#
# Coverage follows the regular-operator spec adapted to a data-movement op:
#   * shape levels: dedicated depth-axis shape sets merged with the shared
#     tu.selected_shapes() levels (quick/all) as self-pairs, bounded so a
#     single input stays <= 1M elements (the output is ~2x the input size);
#   * value ranges: tu.selected_ranges() over small representative shape sets
#     for every supported dtype (the values must round-trip exactly through
#     the depth-axis placement);
#   * edge cases: empty tensors, nan/inf/+-0.0 passthrough, complex inputs,
#     and the .out overload with its alias semantics;
#   * backward: autograd.grad() against the analytic slice-back gradient;
#   * negative: empty TensorList, mismatched non-depth dims, and non-tensor
#     list elements raise on both paths.
_FLOAT_DTYPES = set(utils.ALL_FLOAT_DTYPES)
DSTACK_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Dedicated depth-axis shape sets. dstack views each input as 3-D and
# concatenates along dim 2, so every dim except dim 2 must match while the
# depth dim may vary freely: 1-D -> (1, N, 1), 2-D -> (M, N, 1), 3-D with
# equal/varying depth, a 4-D self-pair, and (in the "all" level) a 5-D case
# whose dim-2 sizes differ (64/96/32) to exercise the "all dims except dim 2
# must match" rule.
if tu.LEVEL == "quick":
    _DSTACK_EXTRA_SHAPE_SETS = [
        [(3,), (3,)],
        [(8, 16, 32), (8, 16, 48)],
    ]
    _DSTACK_RANGE_SHAPE_SETS = [
        [(), ()],
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(4, 5, 6), (4, 5, 7)],
    ]
    _DSTACK_OUT_SHAPE_SETS = [
        [(3,), (3,)],
        [(8, 16, 32), (8, 16, 48)],
    ]
elif tu.LEVEL == "all":
    _DSTACK_EXTRA_SHAPE_SETS = [
        [(3,), (3,)],
        [(3, 33), (3, 33)],
        [(16, 16, 333), (16, 16, 333), (16, 16, 333)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
    ]
else:  # core
    _DSTACK_EXTRA_SHAPE_SETS = [
        [(3,), (3,)],
        [(3, 33), (3, 33)],
        [(16, 16, 333), (16, 16, 333), (16, 16, 333)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
    ]

# Small shape sets for the full value-range sweep (scalar, 1-D, 2-D, and 3-D
# with equal and varying depth).
if tu.LEVEL != "quick":
    _DSTACK_RANGE_SHAPE_SETS = [
        [(), ()],
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(4, 5, 6), (4, 5, 6)],
        [(4, 5, 6), (4, 5, 7)],
    ]

# Empty-tensor shape sets: 1-D, 2-D and 3-D tensors with a zero-size dim.
_DSTACK_EMPTY_SHAPE_SETS = [
    [(0,), (0,)],
    [(2, 0), (2, 0)],
    [(0, 3, 4), (0, 3, 4)],
]

# Small shape sets for the backward test (autograd graph + grad comparison).
_DSTACK_BACKWARD_SHAPE_SETS = [
    [(3,), (3,)],
    [(4, 5), (4, 5)],
    [(4, 5, 6), (4, 5, 7)],
]

# The .out overload runs a representative subset of the shape sets.
if tu.LEVEL != "quick":
    _DSTACK_OUT_SHAPE_SETS = [
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(8, 16, 32), (8, 16, 48)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
    ]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _dstack_shape_sets():
    """Shape-list levels for the main sweep.

    The dedicated depth-axis sets above are merged with the shared shape levels
    (tu.selected_shapes(), quick/all) as self-pairs. Each pair keeps every
    dim except dim 2 identical so the depth-axis concatenation is exercised;
    self-pairs whose single input would exceed 1M elements are skipped because
    the output is ~2x the input size.
    """
    shape_sets = list(_DSTACK_EXTRA_SHAPE_SETS)
    for shape in tu.selected_shapes():
        if _numel(shape) > 2**20:
            continue
        pair = [shape, shape]
        if pair not in shape_sets:
            shape_sets.append(pair)
    return shape_sets


def _dstack_depth(shape):
    """Number of depth slices an input of ``shape`` occupies after atleast_3d
    (its dim 2; 0-dim/1-dim/2-dim inputs get depth 1)."""
    return utils.unsqueeze_tuple(shape, 3)[2]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.dstack callable, (3) None -> the
    # test falls back to the PyTorch reference so it stays runnable before a
    # FlagGems implementation is registered.
    try:
        return flag_gems.testing.resolve_gems_op(
            "dstack", getattr(flag_gems, "dstack", None)
        )
    except LookupError:
        return None


def _resolve_gems_op_out():
    try:
        return flag_gems.testing.resolve_gems_op(
            "dstack.out", getattr(flag_gems, "dstack_out", None)
        )
    except LookupError:
        return None


def _apply_dstack(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.dstack(inp)
    return gems_op(inp)


def _apply_dstack_out(inp, out):
    gems_op = _resolve_gems_op_out()
    if gems_op is None:
        return torch.ops.aten.dstack.out(inp, out=out)
    return gems_op(inp, out=out)


def _assert_dstack_output(res_out, ref_out, dtype):
    # dstack materializes a new contiguous tensor (never an aliasing view).
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _dstack_shape_sets())
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack(shape_set, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype). 0-dim scalars,
    # 1-D, 2-D, 3-D (equal and varying depth), 4-D and (in the "all" level)
    # 5-D to 8-D self-pairs are all covered.
    inp = [tu.make_input(dtype, s, ["-1", "1"]) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    _assert_dstack_output(res_out, ref_out, dtype)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_RANGE_SHAPE_SETS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_value_ranges(shape_set, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate constant ranges) must round-trip
    # exactly through the depth-axis placement. bool ignores the range; the
    # int/bool compare is exact and the float compare uses equal_nan=True.
    inp = [tu.make_input(dtype, s, value_range) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.dstack_out
@pytest.mark.parametrize("shape_set", _DSTACK_OUT_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_out(shape_set, dtype):
    # The .out overload must write into the provided out tensor and return it
    # (alias semantics), matching the aten reference bit-for-bit.
    inp = [tu.make_input(dtype, s, ["-1", "1"]) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_shape = torch.ops.aten.dstack(ref_inp).shape
    ref_out = torch.empty(ref_shape, dtype=dtype, device=ref_inp[0].device)
    ref_ret = torch.ops.aten.dstack.out(ref_inp, out=ref_out)

    out = torch.empty(ref_shape, dtype=dtype, device=inp[0].device)
    res_ret = _apply_dstack_out(inp, out)

    # The .out variant must return the out tensor itself (alias semantics).
    assert res_ret.data_ptr() == out.data_ptr()
    assert ref_ret.data_ptr() == ref_out.data_ptr()
    assert res_ret.shape == ref_ret.shape
    assert res_ret.dtype == ref_ret.dtype
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_ret, ref_ret, dtype)
        utils.gems_assert_close(out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_ret, ref_ret)
        utils.gems_assert_equal(out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_EMPTY_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_empty_inputs(shape_set, dtype):
    # Zero-sized tensors: 1-D (0,), 2-D (2, 0) and 3-D (0, 3, 4) all produce
    # valid (possibly empty) depth-axis concatenations.
    inp = [tu.make_input(dtype, s, ["-1", "1"]) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    _assert_dstack_output(res_out, ref_out, dtype)


@pytest.mark.dstack
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_dstack_nan_inf(dtype):
    # dstack is a pure data-movement op: +inf/-inf/nan/+-0.0 pass through
    # unchanged onto the depth axis (assert_result_close uses equal_nan=True on
    # the float path; 1e30 overflows to inf in fp16/bf16 on both paths
    # identically).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = [values, values]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_dstack_complex(dtype):
    # dstack also supports complex tensors (a pure data-movement op: real and
    # imaginary parts round-trip untouched). One negative-and-positive range
    # per dtype suffices because no arithmetic is performed.
    inp = [
        tu.make_input(dtype, (4, 5, 6), ["-1", "1"]),
        tu.make_input(dtype, (4, 5, 7), ["-1", "1"]),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.dstack_backward
@pytest.mark.parametrize("shape_set", _DSTACK_BACKWARD_SHAPE_SETS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack_backward(shape_set, dtype):
    # dstack = atleast_3d(each input) + cat along dim 2, so grad_i is the slice
    # of grad_out owned by input i, reshaped back to the input's shape (a pure
    # gather, no arithmetic). Validate the autograd reference against that
    # analytic value, then check the candidate forward output and - only when
    # the candidate output is differentiable - its gradient against the
    # reference gradient.
    inp = [tu.make_input(dtype, s, ["-1", "1"]).requires_grad_() for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = utils.to_reference(grad)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    if dtype in (torch.float32, torch.float64):
        offset = 0
        for t, g in zip(ref_inp, ref_in_grads):
            depth = _dstack_depth(t.shape)
            expected = torch.ops.aten.slice(
                ref_grad, 2, offset, offset + depth
            ).reshape(t.shape)
            tu.assert_result_close(g, expected)
            offset += depth

    res_out = _apply_dstack(inp)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
        for res_g, ref_g, src in zip(res_in_grads, ref_in_grads, inp):
            assert res_g.shape == ref_g.shape == src.shape
            tu.assert_result_close(res_g, ref_g)


@pytest.mark.dstack_negative
def test_dstack_empty_list():
    # dstack expects a non-empty TensorList; the candidate must fail too rather
    # than silently return an empty tensor.
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op([])


@pytest.mark.dstack_negative
@pytest.mark.parametrize(
    "shape_set",
    [
        [(2, 3), (4, 3)],  # dim 0 mismatch (2-D inputs: (2, 3, 1) vs (4, 3, 1))
        [(3,), (2, 3)],  # 1-D (1, 3, 1) vs 2-D (2, 3, 1): dim 0 mismatch
        [(4, 5, 6), (4, 7, 6)],  # dim 1 mismatch
    ],
)
def test_dstack_mismatched_shapes(shape_set):
    # All dims except dim 2 must match after the atleast_3d view; mismatched
    # non-depth dims must raise on both paths.
    inp = [tu.make_input(torch.float32, s, ["-1", "1"]) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack(ref_inp)
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(inp)


@pytest.mark.dstack_negative
def test_dstack_rejects_non_tensor():
    # The aten op requires a TensorList of Tensors; a Python float list element
    # must raise on both paths.
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([torch.zeros(2, device=flag_gems.device), 3.14])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op([torch.zeros(2, device=flag_gems.device), 3.14])
