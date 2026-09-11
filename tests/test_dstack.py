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
#   * dtypes -- probed with tu.supported_dtypes; the default single-tensor probe
#     does not apply because dstack takes a TensorList, so a custom probe builds
#     a two-element list. int8/uint8/fp8 are hard requirements and are kept
#     because the active backend supports them (fp8 is compared through the
#     exact device-resident helper since torch.testing cannot compare float8 on
#     CPU);
#   * value ranges -- the full tu.selected_ranges() sweep ([-1,1], [0,1],
#     [-1,0], [0,max], [min,0]) for every supported dtype (the old randn-only
#     value test is migrated onto this framework);
#   * shape levels -- dedicated depth-axis sets merged with the shared shape
#     levels tu.selected_shapes() (quick/all via --quick) as self-pairs,
#     bounded so one input stays <= 2**20 elements (the output is ~n_inputs x
#     the input) and every rank 0..5 is represented;
#   * broadcast -- N/A: dstack has no broadcast dimension, all non-depth dims
#     must match, so the broadcast dimension is skipped;
#   * backward -- autograd.grad() against the analytic slice-back gradient
#     (grad_i is grad_out's slice for input i reshaped to the input shape);
#   * edge cases -- empty tensors, nan/inf/+-0.0 passthrough, complex inputs;
#   * negative -- empty TensorList, mismatched non-depth dims and non-tensor
#     list elements raise on both the reference and the candidate path;
#   * the .out overload is probed invocable on the active backend and is tested
#     with alias (write-into-and-return-out) semantics.
#
# The candidate is resolved through flag_gems.testing.resolve_gems_op(...)
# inside each test (never at import time) so the process-local override
# installed by KernelGen wins. When neither an override nor a native
# implementation is registered yet, the tests fall back to the PyTorch
# reference so the file stays runnable standalone.

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

# The spec's required dtype list first (int8 / uint8 / fp8 are hard
# requirements when the backend supports them), then the shared float/int/bool
# sets. The probe below removes anything the active backend cannot handle.
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


def _probe_dstack(operator, dtype):
    """Backend-support probe for the Tensor[] op.

    The shared default probe passes a single tensor, but dstack requires a
    TensorList, so build a two-element list and compare the result dtype. Any
    exception means the active backend cannot run dstack for that dtype.
    """
    del operator
    try:
        x = tu.make_input(dtype, (4,), ["0", "1"])
        out = torch.ops.aten.dstack([x, x])
    except Exception:
        return False
    return out.dtype == dtype


DSTACK_DTYPES = tu.supported_dtypes(
    "dstack", candidates=_DTYPE_CANDIDATES, probe=_probe_dstack
)
if not DSTACK_DTYPES:
    # Never collect zero cases: fall back to the full candidate list so a
    # failed/absent probe never silently drops the spec-required int8/uint8/fp8
    # dtypes. (_DTYPE_CANDIDATES also carries the complex dtypes; those are
    # additionally covered by the dedicated DSTACK_COMPLEX_DTYPES cases below,
    # so the overlap is harmless.)
    DSTACK_DTYPES = list(_DTYPE_CANDIDATES)

# Complex dtypes are covered as their own case (make_tensor fills the real and
# imaginary parts); they are probed separately for the same reason as above.
DSTACK_COMPLEX_DTYPES = [
    dtype for dtype in utils.COMPLEX_DTYPES if _probe_dstack("dstack", dtype)
]
if not DSTACK_COMPLEX_DTYPES:
    DSTACK_COMPLEX_DTYPES = list(utils.COMPLEX_DTYPES)

_MAIN_RANGE = ["-1", "1"]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


# Dedicated depth-axis shape sets. dstack views each input as 3-D and
# concatenates along dim 2, so every dim except dim 2 must match while the
# depth dim may vary freely: 1-D -> (1,N,1), 2-D -> (M,N,1), 3-D with
# equal/varying depth, a 4-D self-pair, and (in "all") a 5-D case whose dim-2
# sizes differ (64/96/32) to exercise the "all dims except dim 2 must match"
# rule.
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
    (tu.selected_shapes(), quick/all) as self-pairs. Self-pairs whose single
    input would exceed 2**20 elements are skipped because the output is
    ~n_inputs x the input size.
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
    """Depth (dim-2 extent) an input of ``shape`` occupies after atleast_3d;
    0-dim/1-dim/2-dim inputs get depth 1, ndim >= 3 inputs keep their dim 2."""
    return utils.unsqueeze_tuple(shape, 3)[2]


_DTYPE_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in DSTACK_DTYPES
    for value_range in tu.selected_ranges()
]


def _resolve_named_gems_op(name):
    """Resolve one operator/overload name through resolve_gems_op.

    Resolution order: (1) the process-local override installed by KernelGen,
    (2) the direct flag_gems callable for that name, (3) None -> the caller
    falls back to the PyTorch reference so the file is runnable standalone.
    """
    default = getattr(flag_gems, name.replace(".", "_"), None)
    if default is None:
        default = getattr(flag_gems, name, None)
    try:
        return flag_gems.testing.resolve_gems_op(name, default)
    except LookupError:
        return None


def _resolve_gems_op():
    return _resolve_named_gems_op("dstack")


def _resolve_gems_op_out():
    # The harness may register the out overload as "dstack.out" or
    # "dstack_out"; try both and fall back to the reference when neither
    # exists (the main "dstack" callable is deliberately not reused here).
    for name in ("dstack.out", "dstack_out"):
        op = _resolve_named_gems_op(name)
        if op is not None:
            return op
    return None


def _apply_dstack(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        return torch.ops.aten.dstack(inp)
    return gems_op(inp)


def _apply_dstack_out(inp, out):
    gems_op = _resolve_gems_op_out()
    if gems_op is None:
        return torch.ops.aten.dstack.out(inp, out=out)
    return gems_op(inp, out=out)


def _assert_values(res_out, ref_out, dtype):
    """Compare values: fp8 through the exact device helper (torch.testing has
    no CPU fp8 comparison; float8 values are exact in float32), everything else
    through the tolerance-aware value-range helper (exact for int/bool)."""
    if dtype in _FP8_DTYPES:
        utils.gems_assert_equal(res_out.to(torch.float32), ref_out.to(torch.float32))
    else:
        tu.assert_result_close(res_out, ref_out)


def _assert_dstack_output(res_out, ref_out, dtype):
    # dstack materializes a new contiguous tensor (never an aliasing view).
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    _assert_values(res_out, ref_out, dtype)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _dstack_shape_sets())
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack(shape_set, dtype):
    # Shape levels x every supported dtype, with values from the shared
    # non-degenerate [-1,1] range (tu.make_input clamps the negative bound for
    # dtypes that cannot represent it).
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    _assert_dstack_output(res_out, ref_out, dtype)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_RANGE_SHAPE_SETS)
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test_dstack_value_ranges(shape_set, dtype, value_range):
    # The op never transforms the stored values, so the full spec range sweep
    # (0/max/min and the degenerate constant ranges included) must round-trip
    # exactly through the depth-axis placement.
    inp = [tu.make_input(dtype, s, value_range) for s in shape_set]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    _assert_values(res_out, ref_out, dtype)


@pytest.mark.dstack_out
@pytest.mark.parametrize("shape_set", _DSTACK_OUT_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_out(shape_set, dtype):
    # The .out overload must write into the provided out tensor and return it
    # (alias semantics), matching the aten reference bit-for-bit.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
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
    _assert_values(res_ret, ref_ret, dtype)
    _assert_values(out, ref_out, dtype)


@pytest.mark.dstack
@pytest.mark.parametrize("shape_set", _DSTACK_EMPTY_SHAPE_SETS)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_empty_inputs(shape_set, dtype):
    # Zero-sized tensors: 1-D (0,), 2-D (2, 0) and 3-D (0, 3, 4) all produce
    # valid (possibly empty) depth-axis concatenations.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE) for s in shape_set]
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
@pytest.mark.parametrize("dtype", DSTACK_COMPLEX_DTYPES)
def test_dstack_complex(dtype):
    # dstack also supports complex tensors (a pure data-movement op: real and
    # imaginary parts round-trip untouched). One negative-and-positive range
    # per dtype suffices because no arithmetic is performed.
    inp = [
        tu.make_input(dtype, (4, 5, 6), _MAIN_RANGE),
        tu.make_input(dtype, (4, 5, 7), _MAIN_RANGE),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    _assert_dstack_output(res_out, ref_out, dtype)


@pytest.mark.dstack_backward
@pytest.mark.parametrize("shape_set", _DSTACK_BACKWARD_SHAPE_SETS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack_backward(shape_set, dtype):
    # dstack = atleast_3d(each input) + cat along dim 2, so grad_i is the slice
    # of grad_out owned by input i, reshaped back to the input's shape (a pure
    # gather, no arithmetic). Validate the autograd reference against that
    # analytic value, then check the candidate forward and - only when the
    # candidate output is differentiable - its gradient against the reference.
    inp = [tu.make_input(dtype, s, _MAIN_RANGE).requires_grad_() for s in shape_set]
    ref_inp = [utils.to_reference(t.detach().clone()).requires_grad_() for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    grad = tu.make_input(dtype, ref_out.shape, _MAIN_RANGE)
    ref_grad = utils.to_reference(grad)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    offset = 0
    for t, g in zip(ref_inp, ref_in_grads):
        depth = _dstack_depth(t.shape)
        expected = torch.ops.aten.slice(ref_grad, 2, offset, offset + depth).reshape(
            t.shape
        )
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
        [(2, 3), (4, 3)],  # dim 0 mismatch: (2,3,1) vs (4,3,1)
        [(3,), (2, 3)],  # 1-D (1,3,1) vs 2-D (2,3,1): dim 0 mismatch
        [(4, 5, 6), (4, 7, 6)],  # dim 1 mismatch
    ],
)
def test_dstack_mismatched_shapes(shape_set):
    # All dims except dim 2 must match after the atleast_3d view; mismatched
    # non-depth dims must raise on both paths.
    inp = [tu.make_input(torch.float32, s, _MAIN_RANGE) for s in shape_set]
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
