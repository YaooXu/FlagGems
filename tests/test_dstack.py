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

# aten::dstack(Tensor[] tensors) -> Tensor views every input as 3-D
# (atleast_3d: 0-dim -> (1,1,1), 1-dim -> (1,N,1), 2-dim -> (M,N,1), ndim >= 3
# kept as-is) and concatenates the results along the new depth axis (dim 2).
# Every dim except dim 2 must match across inputs; the depth dim may vary per
# input. It is a pure data-movement op (no arithmetic), so stored values
# round-trip unchanged for every storage dtype aten supports (int8/uint8/fp8/
# fp16/bf16/fp32/fp64/int16/int32/int64/bool/complex), with nan/inf/-inf/+-0.0
# passing through untouched.
#
# Coverage follows the regular-operator test spec adapted to a Tensor[] op:
#   * dtypes -- the required set includes int8/uint8/FP8, plus the shared
#     float/int/bool families (FP8 is compared through the
#     exact device-resident helper since torch.testing cannot compare float8 on
#     CPU);
#   * value ranges -- the full tu.selected_ranges() sweep ([-1,1], [0,1],
#     [-1,0], [0,max], [min,0]) for every supported dtype (the old randn-only
#     value test is migrated onto this framework);
#   * shape levels -- dedicated depth-axis sets merged with the shared shape
#     levels tu.selected_shapes() (quick/default via --quick) as self-pairs,
#     including every required shape from rank 0 through rank 5;
#   * broadcast -- N/A: dstack has no broadcast dimension, all non-depth dims
#     must match, so the broadcast dimension is skipped;
#   * backward -- autograd.grad() against the analytic slice-back gradient
#     (grad_i is grad_out's slice for input i reshaped to the input shape);
#   * edge cases -- empty tensors, nan/inf/+-0.0 passthrough, complex inputs;
#   * negative -- empty TensorList, mismatched non-depth dims and non-tensor
#     list elements raise on both the reference and the candidate path;
#   * the .out overload is tested
#     with alias (write-into-and-return-out) semantics.
#
# The candidate is resolved through flag_gems.testing.resolve_gems_op(...)
# inside each test (never at import time) so the process-local override
# installed by KernelGen wins. Resolution raises LookupError when no override
# and no native implementation is registered; that error is not caught, so a
# test never passes by running the PyTorch reference instead of the candidate.

_FP8_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

# Required dtypes first, then the shared float/int/bool/complex sets.
_DTYPE_CANDIDATES = []
for _dtype in (
    list(tu.REQUIRED_DTYPES)
    + list(utils.ALL_FLOAT_DTYPES)
    + list(utils.ALL_INT_DTYPES)
    + list(utils.BOOL_TYPES)
    + list(utils.COMPLEX_DTYPES)
):
    if _dtype not in _DTYPE_CANDIDATES:
        _DTYPE_CANDIDATES.append(_dtype)


DSTACK_DTYPES = list(_DTYPE_CANDIDATES)

# Complex dtypes are covered as their own case (make_tensor fills the real and
# imaginary parts).
DSTACK_COMPLEX_DTYPES = list(utils.COMPLEX_DTYPES)

_MAIN_RANGE = ["-1", "1"]


# Dedicated depth-axis shape sets. dstack views each input as 3-D and
# concatenates along dim 2, so every dim except dim 2 must match while the
# depth dim may vary freely: 1-D -> (1,N,1), 2-D -> (M,N,1), 3-D with
# equal/varying depth, a 4-D self-pair, and (in "all") a 5-D case whose dim-2
# sizes differ (64/96/32) to exercise the "all dims except dim 2 must match"
# rule.
if tu.QUICK_MODE:
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
else:  # "all"
    _DSTACK_EXTRA_SHAPE_SETS = [
        [(3,), (3,)],
        [(3, 33), (3, 33)],
        [(16, 16, 333), (16, 16, 333), (16, 16, 333)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
    ]
    _DSTACK_RANGE_SHAPE_SETS = [
        [(), ()],
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(4, 5, 6), (4, 5, 6)],
        [(4, 5, 6), (4, 5, 7)],
    ]
    _DSTACK_OUT_SHAPE_SETS = [
        [(3,), (3,)],
        [(4, 5), (4, 5)],
        [(8, 16, 32), (8, 16, 48)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
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


def _dstack_shape_sets():
    """Shape-list levels for the main sweep.

    The dedicated depth-axis sets are merged with the shared shape levels
    (tu.selected_shapes(), quick/default) as self-pairs.
    """
    shape_sets = list(_DSTACK_EXTRA_SHAPE_SETS)
    for shape in tu.selected_shapes():
        pair = [shape, shape]
        if pair not in shape_sets:
            shape_sets.append(pair)
    return shape_sets


_DTYPE_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in DSTACK_DTYPES
    for value_range in tu.selected_ranges()
]


def _resolve_named_gems_op(name):
    """Resolve one operator name through resolve_gems_op.

    Resolution order: (1) the process-local override installed by KernelGen,
    (2) the direct flag_gems callable for that name. ``LookupError`` is not
    caught here: the caller must not substitute the PyTorch reference for a
    missing candidate.
    """
    default = getattr(flag_gems, name.replace(".", "_"), None)
    if default is None:
        default = getattr(flag_gems, name, None)
    return flag_gems.testing.resolve_gems_op(name, default)


def _resolve_gems_op():
    return _resolve_named_gems_op("dstack")


def _assert_dstack_output(res_out, ref_out):
    # dstack materializes a new contiguous tensor (never an aliasing view).
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _dstack_shape_sets())
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack(shape_set, dtype):
    # Shape levels x every supported dtype, with values from the shared
    # non-degenerate [-1,1] range (tu.make_input clamps the negative bound for
    # dtypes that cannot represent it).
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_RANGE_SHAPE_SETS)
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test_dstack_value_ranges(shape_set, dtype, value_range):
    # The op never transforms the stored values, so the full spec range sweep
    # (0/max/min and the degenerate constant ranges included) must round-trip
    # exactly through the depth-axis placement.
    inp = [tu.make_input(dtype, s, value_range) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack_out
@pytest.mark.parametrize("shape_set", _DSTACK_OUT_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_out(shape_set, dtype):
    # The .out overload must write into the provided out tensor and return it
    # (alias semantics), matching the aten reference bit-for-bit.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_shape = torch.ops.aten.dstack(ref_inp).shape
    ref_out = torch.empty(ref_shape, dtype=dtype, device=ref_inp[0].device)
    ref_ret = torch.ops.aten.dstack.out(ref_inp, out=ref_out)

    out = torch.empty(ref_shape, dtype=dtype, device=inp[0].device)
    res_ret = _resolve_gems_op()(inp, out=out)

    # The .out variant must return the out tensor itself (alias semantics).
    assert res_ret.data_ptr() == out.data_ptr()
    tu.assert_result_equal(res_ret, ref_ret)
    tu.assert_result_equal(out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_EMPTY_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_empty_inputs(shape_set, dtype):
    # Zero-sized tensors: 1-D (0,), 2-D (2, 0) and 3-D (0, 3, 4) all produce
    # valid (possibly empty) depth-axis concatenations.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(DSTACK_DTYPES))
)
def test_dstack_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    inp = [values, values]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("dtype", DSTACK_COMPLEX_DTYPES)
def test_dstack_complex(dtype):
    # dstack also supports complex tensors (a pure data-movement op: real and
    # imaginary parts round-trip untouched). One negative-and-positive range
    # per dtype suffices because no arithmetic is performed.
    inp = [
        tu.make_input(dtype, (4, 5, 6), _MAIN_RANGE),
        tu.make_input(dtype, (4, 5, 7), _MAIN_RANGE),
    ]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_dstack_output(res_out, ref_out)


@pytest.mark.dstack_backward
@pytest.mark.parametrize("shape_set", _DSTACK_BACKWARD_SHAPE_SETS)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [d for d in DSTACK_DTYPES if d.is_floating_point or d.is_complex]
    ),
)
def test_dstack_backward(shape_set, dtype):
    # Backward slices and reshapes the upstream gradient without arithmetic.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE).requires_grad_() for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    grad = tu.make_input(dtype, ref_out.shape, _MAIN_RANGE)
    ref_grad = tu.to_reference(grad)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = _resolve_gems_op()(inp)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for res_g, ref_g in zip(res_in_grads, ref_in_grads):
        tu.assert_result_equal(res_g, ref_g)


@pytest.mark.dstack_negative
def test_dstack_empty_list():
    # dstack expects a non-empty TensorList; the candidate must fail too rather
    # than silently return an empty tensor.
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([])
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op([])


@pytest.mark.dstack_negative
@pytest.mark.parametrize(
    "shape_set",
    [
        [(2, 3), (4, 3)],  # dim 0 mismatch: (2,3,1) vs (4,3,1)
        [(3,), (2, 3)],  # 1-D (1,3,1) vs 2-D (2,3,1): dim 0 mismatch
        [(4, 5, 6), (4, 7, 6)],  # dim 1 mismatch
    ],
)
def test_dstack_mismatched_shapes(shape_set):
    # All dims except dim 2 must match after the atleast_3d view; mismatched
    # non-depth dims must raise on both paths.
    inp = [tu.make_input(torch.float32, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [tu.to_reference(t) for t in inp]

    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack(ref_inp)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(inp)


@pytest.mark.dstack_negative
def test_dstack_rejects_non_tensor():
    # The aten op requires a TensorList of Tensors; a Python float list element
    # must raise on both paths.
    with pytest.raises(RuntimeError):
        torch.ops.aten.dstack([torch.zeros(2, device=flag_gems.device), 3.14])
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op([torch.zeros(2, device=flag_gems.device), 3.14])
