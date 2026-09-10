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

# ``_remove_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._remove_batch_dim`` and ``-m
# _remove_batch_dim`` both work.
setattr(
    pytest.mark,
    "_remove_batch_dim",
    MarkDecorator(Mark("_remove_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_remove_batch_dim(Tensor self, int level, SymInt batch_size, int out_dim)
# is the functorch/vmap unwrap primitive. On a plain (non-batched) tensor it is
# exactly ``self.expand(sizes)`` where ``sizes`` is ``self.shape`` with
# ``batch_size`` inserted at position ``out_dim``: a batch dimension of size
# ``batch_size`` is created at ``out_dim`` and broadcast along the whole tensor
# (the inserted dim gets stride 0). ``level`` is only vmap bookkeeping and never
# affects the result.
#
# Broadcast is valid when every dim of ``self`` (aligned to the trailing dims of
# the target) either equals the corresponding target dim or is 1, so the
# (shape, out_dim, batch_size) cases below are chosen so that expand always
# succeeds. Together they cover ranks 0-5, every valid out_dim class (front,
# middle, end), batch_size matching the adjacent dim, and size-1 broadcast. The
# operator performs no arithmetic (it returns a zero-copy view), so every
# storage dtype is supported and element counts for the case grid stay bounded;
# the value-range sweep additionally walks the spec's seven shape levels with a
# non-broadcast insert (batch_size=1 at out_dim=0) plus a broadcast case.
#
# Coverage map (regular-operator spec):
#   * value ranges: the five spec ranges (``tu.selected_ranges()``) over every
#     shape level (``tu.selected_shapes()``) and every supported dtype;
#   * shapes: the spec's seven shape levels (0-5 dims), driven by pytest
#     ``--quick`` through ``tu.selected_shapes()``;
#   * dtypes: fp16/bf16/fp32/fp64, int16/int32/int64, int8/uint8,
#     float8_e4m3fn/float8_e5m2 and bool (the op is a view, so all are accepted);
#   * non-contiguous inputs (strides and storage offset must survive);
#   * backward: expand is differentiable, so autograd.grad must match aten's
#     (including the sum-reduction over every broadcast dim);
#   * nan/inf/-inf and signed zeros pass through the broadcast untouched;
#   * negative: non-broadcastable / negative batch_size and non-tensor inputs
#     are rejected (matching aten's own validation).
#
# No broadcast-vs-binary dimension applies (the op is unary and has no second
# tensor operand); "broadcast" here means the data broadcast performed by the
# inserted batch dim, which the case grid and the value sweep both exercise.

# The op is a pure metadata view, so it accepts every storage dtype. int8/uint8
# and the two float8 flavours are added on top of the shared dtype sets to meet
# the required dtype coverage.
_SPECIAL_VALUE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_REMOVE_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + _SPECIAL_VALUE_DTYPES
)

# float8 comparison is upcast losslessly to float32 first (flag_gems'
# assert_close keys its tolerance table on the dtype and expects a supported
# floating dtype); the view is bit-exact so the upcast changes nothing.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# The spec's [-1, 0] range has no representable unsigned counterpart (it
# collapses to the empty [0, 0) interval), so it is skipped for uint8.
_UNSIGNED_INT_DTYPES = (torch.uint8,)

# (shape, out_dim, batch_size) grid. Every combination is a valid expand:
# inserting batch_size at out_dim and aligning self to the trailing dims keeps
# all non-broadcast dims equal and never expands a size-1 dim to a different
# extent. Ranks 0-5, out_dim at the front/middle/end, batch_size matching the
# adjacent dim, size-1 broadcast, the spec shape levels, and the empty/1-element
# degenerates are all represented.
_REMOVE_BATCH_DIM_CASES = [
    ((), 0, 7),  # rank-0: batch becomes the only dim
    ((16,), 0, 7),  # rank-1, out_dim at the front
    ((16,), 1, 16),  # rank-1, out_dim at the end (batch == s0)
    ((1,), 0, 5),  # rank-1, size-1 dim at the front
    ((1,), 1, 9),  # rank-1, size-1 dim broadcast at the end
    ((256,), 0, 11),  # rank-1 regular 1-D shape, front
    ((256,), 1, 256),  # rank-1, out_dim at the end (batch == s0)
    ((64, 32), 0, 13),  # rank-2, out_dim at the front
    ((64, 32), 1, 64),  # rank-2, batch matches dim0
    ((1, 32), 1, 7),  # rank-2, size-1 broadcast at dim0
    ((1024, 1024), 0, 1),  # rank-2 regular 2-D shape, batch size 1
    ((2, 19, 7), 0, 5),  # rank-3, out_dim at the front
    ((2, 19, 7), 1, 2),  # rank-3, batch matches dim0
    ((1, 19, 7), 1, 5),  # rank-3, size-1 broadcast at dim0
    ((1, 19, 7), 2, 19),  # rank-3, middle out_dim, size-1 dim0
    ((4, 4, 16), 2, 4),  # rank-3, middle out_dim, adjacent dims equal
    ((20, 320, 15), 0, 1),  # rank-3 spec shape, out_dim at the front
    ((20, 320, 15), 1, 20),  # rank-3 spec shape, batch matches dim0
    ((4, 8, 16, 32), 0, 9),  # rank-4, out_dim at the front
    ((4, 8, 16, 32), 1, 4),  # rank-4, batch matches dim0
    ((8, 8, 8, 32), 2, 8),  # rank-4, middle out_dim
    ((1, 8, 16, 32), 2, 8),  # rank-4, middle out_dim, size-1 dim0
    ((16, 128, 64, 60), 0, 1),  # rank-4 spec shape, batch size 1
    ((16, 7, 57, 32, 29), 0, 1),  # rank-5, out_dim at the front
    ((1, 7, 57, 32, 29), 1, 11),  # rank-5, size-1 dim0 broadcast
]

# Backward of expand reduces the gradient over every broadcast (stride-0) dim,
# so the cases below cover: a batch dim matching dim0 (sum over the new batch
# dim), a front batch dim (sum over the interpolated batch dim), and a size-1
# dim broadcast (sum over the expanded size-1 dim and the new batch dim).
_BACKWARD_CASES = [
    ((2, 19, 7), 1, 2),
    ((4, 8, 16, 32), 0, 9),
    ((1, 19, 7), 2, 19),
]

# Non-contiguous (strided) inputs: slicing on the last dim keeps a non-unit
# stride so the view must preserve strides and storage offset.
_NON_CONTIGUOUS_CASES = [
    ((8, 16, 32), 1, 8),
    ((8, 8, 16, 32), 2, 8),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order is:
    # (1) override, (2) the direct flag_gems._remove_batch_dim callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_remove_batch_dim", getattr(flag_gems, "_remove_batch_dim", None)
    )


def _expected_shape(shape, out_dim, batch_size):
    sizes = list(shape)
    sizes.insert(out_dim, batch_size)
    return tuple(sizes)


def _as_comparable(t):
    if t.dtype in _FP8_DTYPES:
        return t.to(torch.float32)
    return t


def _assert_values_close(res, ref):
    # int/bool must match bit-exactly; floating point uses the shared tolerance
    # with equal_nan=True (the broadcast view repeats the stored values exactly).
    res_c = _as_comparable(res)
    ref_c = _as_comparable(ref)
    if res_c.dtype == torch.bool or not res_c.is_floating_point():
        utils.gems_assert_equal(res_c, ref_c)
    else:
        utils.gems_assert_close(res_c, ref_c, res_c.dtype, equal_nan=True)


def _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype):
    assert res_out.shape == ref_out.shape == _expected_shape(shape, out_dim, batch_size)
    assert res_out.dtype == ref_out.dtype == dtype
    _assert_values_close(res_out, ref_out)


def _dtype_range_pairs():
    pairs = []
    for dtype in _REMOVE_BATCH_DIM_DTYPES:
        for value_range in tu.selected_ranges():
            if dtype in _UNSIGNED_INT_DTYPES and value_range == ["-1", "0"]:
                continue
            pairs.append((dtype, value_range))
    return pairs


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _REMOVE_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim(shape, out_dim, batch_size, level, dtype):
    # Values are irrelevant to the view itself (a representative [-1, 1] range
    # keeps every storage dtype valid); the dedicated value-range test below
    # sweeps the full spec ranges. ``level`` must never change the result.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype, value_range", _dtype_range_pairs())
def test__remove_batch_dim_value_ranges(shape, dtype, value_range):
    # Every spec shape level x every spec value range x every supported dtype.
    # batch_size=1 at out_dim=0 is a valid insert for any shape (it only adds a
    # leading size-1 dim), so the shape dimension can be swept without changing
    # shape rank semantics; the broadcast case below covers true data broadcast.
    out_dim, batch_size = 0, 1
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, 0, batch_size, out_dim)

    _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype, value_range", _dtype_range_pairs())
def test__remove_batch_dim_value_ranges_broadcast(dtype, value_range):
    # A true broadcast insert: batch_size=2 at out_dim=1 of (2, 19, 7) targets
    # (2, 2, 19, 7), so the whole tensor is repeated along the new stride-0 batch
    # dim and every value range must round-trip exactly.
    shape, out_dim, batch_size = (2, 19, 7), 1, 2
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, 0, batch_size, out_dim)

    _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_non_contiguous(shape, out_dim, batch_size, level, dtype):
    # The broadcast view must preserve the strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout. The sliced shape is
    # what out_dim/batch_size must be valid for.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, inp.shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__remove_batch_dim_nan_inf(dtype):
    # The view never performs arithmetic, so nan/inf/-inf and signed zeros pass
    # through the broadcast untouched (equal_nan=True in the float comparison).
    vals = [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        1.5,
        -2.5,
        1e30,
        -1e30,
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device).reshape(3, 3)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, 4, 0)
    res_out = _resolve_gems_op()(inp, 0, 4, 0)

    _assert_values_close(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__remove_batch_dim_backward(shape, out_dim, batch_size, dtype):
    # expand is differentiable: the gradient of the broadcast view is the
    # sum-reduction of grad_output over every broadcast (stride-0) dim. The
    # candidate gradient must match aten's, including the reduction.
    level = 0
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    out_shape = _expected_shape(shape, out_dim, batch_size)
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_grad_out = utils.to_reference(grad_out)

    inp.requires_grad_(True)
    ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    res_grad = torch.autograd.grad(res_out, inp, grad_out)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)[0]

    assert res_grad.shape == ref_grad.shape == inp.shape
    _assert_values_close(res_grad, ref_grad)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_non_broadcastable_batch_size():
    # Inserting batch_size=3 at out_dim=1 into (2, 19, 7) targets (2, 3, 19, 7):
    # dim 0 of self is 2 and can neither equal 3 nor broadcast from 1, so aten
    # raises RuntimeError and the candidate must too.
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, 3, 1)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 0, 3, 1)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_negative_batch_size():
    # A negative batch_size is rejected by expand; the candidate must reproduce
    # the validation.
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, -1, 0)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 0, -1, 0)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_non_tensor():
    # The aten schema requires a Tensor for ``self``; a Python scalar hits the
    # argument-check path and raises. The candidate must fail too rather than
    # silently return a bogus view.
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(3.14, 0, 1, 0)
    with pytest.raises((RuntimeError, TypeError, ValueError, AttributeError)):
        _resolve_gems_op()(3.14, 0, 1, 0)
