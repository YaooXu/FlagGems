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

# ``_dim_arange`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._dim_arange`` and ``-m _dim_arange`` both
# work.
setattr(
    pytest.mark,
    "_dim_arange",
    MarkDecorator(Mark("_dim_arange", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dim_arange(Tensor like, int dim) -> Tensor builds a fresh 1-D int64
# tensor of length like.size(dim) holding the values [0, 1, ..., size(dim)-1].
# Only the shape and device of ``like`` are consulted; its values, dtype, strides
# and layout never influence the result. The output is always a fresh (non-view,
# non-alias) int64 tensor on the same device as ``like``. 0-D ``like`` raises
# IndexError for every dim, so the scalar shape is excluded from the valid
# workloads below and covered by the negative tests instead.
#
# Regular-operator spec adaptation notes:
# - Value ranges: the input values are semantically irrelevant, but every range
#   from the spec (tu.selected_ranges()) is still exercised to prove the
#   deterministic arange result is produced for arbitrary storage contents.
# - Shapes: tu.selected_shapes() (the spec's seven shapes, 0-D excluded) plus a
#   few small extra ranks; every valid dim is exercised in both the positive and
#   negative indexing conventions, which aten normalizes identically.
# - Broadcast: N/A -- the op takes a single ``like`` tensor.
# - Backward: N/A -- the output is an int64 index tensor with no autograd
#   support. A dedicated case asserts the result carries no grad_fn.
# - nan/inf: covered by a dedicated case (non-finite storage values are ignored;
#   equal_nan semantics do not apply to the int64 output).

# Spec shape levels via tests/test_utils.py, with a couple of small extra ranks
# for extra dim-convention coverage. The 0-D scalar is dropped: it has no valid
# ``dim`` (see the negative tests below).
_EXTRA_SHAPES = [(1,), (5, 3), (2, 3, 4)]
_DIM_ARANGE_SHAPES = []
for _shape in list(tu.selected_shapes()) + _EXTRA_SHAPES:
    if len(_shape) > 0 and _shape not in _DIM_ARANGE_SHAPES:
        _DIM_ARANGE_SHAPES.append(_shape)
_DIM_ARANGE_CASES = [
    (shape, dim)
    for shape in _DIM_ARANGE_SHAPES
    for dim in range(-len(shape), len(shape))
]

# The op ignores the values *and* the dtype of ``like`` -- it only reads its
# shape/device -- so the spec's full required dtype set (int8, uint8, fp8,
# fp32/bf16/fp16, int32/int64, bool) must be covered wherever the runtime can
# actually build such a tensor and call the op. Probe instead of guessing.
_DTYPE_CANDIDATES = []
for _dtype in (
    list(tu.REQUIRED_DTYPES)
    + utils.ALL_INT_DTYPES
    + utils.FLOAT_DTYPES
    + utils.BOOL_TYPES
):
    if _dtype not in _DTYPE_CANDIDATES:
        _DTYPE_CANDIDATES.append(_dtype)


def _probe_like_dtype(dtype):
    try:
        probe = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
        torch.ops.aten._dim_arange(probe, 0)
    except Exception:
        return False
    return True


_DIM_ARANGE_INPUT_DTYPES = [d for d in _DTYPE_CANDIDATES if _probe_like_dtype(d)]

# Non-contiguous views of a (4, 8, 6) base: (view_fn, logical_shape, dim,
# expected_len). The logical shape, not the storage, must drive the result.
_VIEW_CASES = [
    (lambda b: b.transpose(0, 1), (8, 4, 6), 0, 8),
    (lambda b: b.transpose(0, 1), (8, 4, 6), 1, 4),
    (lambda b: b[0:3, 2:7, 1], (3, 5), 1, 5),
    (lambda b: b.narrow(1, 1, 5), (4, 5, 6), 1, 5),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. ``flag_gems._dim_arange``
    # may not be registered yet, so getattr supplies a safe default and
    # resolve_gems_op falls back to the package namespace before raising.
    return flag_gems.testing.resolve_gems_op(
        "_dim_arange", getattr(flag_gems, "_dim_arange", None)
    )


def _make_like(dtype, shape, value_range):
    # Unsigned dtypes cannot represent the negative bounds of some spec ranges;
    # tu.make_input would clamp the low bound to 0 and fail on the resulting
    # empty interval. The values are irrelevant here, so snap the range to the
    # representable part and keep the (shape, dim) semantics under test.
    if dtype == torch.uint8 and tu.resolve_bound(value_range[0], dtype) < 0:
        value_range = ["0", value_range[1] if value_range[1] != "-1" else "1"]
    return tu.make_input(dtype, shape, value_range)


def _assert_arange_result(res_out, ref_out, inp, expected_len):
    # The result is a fresh 1-D int64 tensor on the ``like`` device holding
    # [0, ..., size(dim)-1]; it is never a view/alias of ``like``.
    assert tuple(res_out.shape) == tuple(ref_out.shape) == (expected_len,)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == inp.device
    assert ref_out.device == inp.device or ref_out.device == torch.device("cpu")
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._dim_arange
@pytest.mark.parametrize("shape, dim", _DIM_ARANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_value_ranges(shape, dim, value_range, dtype):
    # The result must be the deterministic arange(like.size(dim)) no matter
    # what values the storage holds, so every range from the regular-operator
    # spec is exercised here (this doubles as the value-range migration of the
    # original randn-based workload).
    inp = _make_like(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = _resolve_gems_op()(inp, dim)

    _assert_arange_result(res_out, ref_out, inp, shape[dim])


@pytest.mark._dim_arange
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_ARANGE_INPUT_DTYPES)
def test__dim_arange_non_contiguous(view_case, value_range, dtype):
    # _dim_arange must work on any tensor layout; only the logical shape is
    # consulted, never the storage.
    view_fn, expected_shape, dim, expected_len = view_case
    base = _make_like(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert tuple(inp.shape) == expected_shape
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, dim)
    res_out = _resolve_gems_op()(inp, dim)

    _assert_arange_result(res_out, ref_out, inp, expected_len)


@pytest.mark._dim_arange
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__dim_arange_nan_inf(dtype):
    # nan/inf are ordinary storage values for this op and must be ignored: the
    # result is still the deterministic arange sequence over the selected dim.
    inp = _make_like(dtype, (4, 8, 6), ["-1", "1"]).clone()
    inp[0, :, 0] = float("inf")
    inp[1, :, 1] = float("-inf")
    inp[2, :, 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 1)
    res_out = _resolve_gems_op()(inp, 1)

    _assert_arange_result(res_out, ref_out, inp, 8)


@pytest.mark._dim_arange
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__dim_arange_no_autograd(dtype):
    # Backward is N/A: the int64 index output is not differentiable, so the
    # result must never carry a grad_fn (and the op must not mutate ``like``).
    inp = _make_like(dtype, (3, 5), ["-1", "1"])
    before = inp.clone().detach()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dim_arange(ref_inp, 1)
    res_out = _resolve_gems_op()(inp, 1)

    _assert_arange_result(res_out, ref_out, inp, 5)
    assert res_out.grad_fn is None
    assert not res_out.requires_grad
    utils.gems_assert_equal(inp, before)


@pytest.mark._dim_arange
def test__dim_arange_rejects_out_of_range_dim():
    # dim must satisfy -like.dim() <= dim < like.dim(); both the positive and
    # the negative out-of-range bounds must raise like aten does.
    inp = _make_like(torch.float32, (3, 5), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 2)
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, -3)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, 2)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, -3)


@pytest.mark._dim_arange
def test__dim_arange_rejects_zero_dim_like():
    # 0-D ``like`` has no dims to arange over; aten raises IndexError for any dim.
    inp = _make_like(torch.float32, (), ["-1", "1"])
    with pytest.raises(IndexError):
        torch.ops.aten._dim_arange(inp, 0)
    with pytest.raises((IndexError, RuntimeError)):
        _resolve_gems_op()(inp, 0)


@pytest.mark._dim_arange
def test__dim_arange_rejects_non_integer_dim():
    # The schema requires an int ``dim``; a float is rejected at binding time.
    inp = _make_like(torch.float32, (4,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._dim_arange(inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp, 1.5)
