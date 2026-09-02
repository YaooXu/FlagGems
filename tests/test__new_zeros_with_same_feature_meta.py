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

import math
import os
import sys

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path and
# re-point the ``tests`` package at this file's directory before importing the
# helpers (under ``--import-mode=importlib`` pytest does not prepend the
# checkout root automatically).
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

# ``_new_zeros_with_same_feature_meta`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._new_zeros_with_same_feature_meta`` and ``-m
# _new_zeros_with_same_feature_meta`` both work.
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta_out",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta_out", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_new_zeros_with_same_feature_meta(Tensor self, Tensor other, *, int
# self_num_batch_dims=0) -> Tensor is a pure allocation helper (used by
# torch.distributions.Independent.expand): it returns a zero tensor whose shape
# is ``self.shape[:self_num_batch_dims] + other.shape`` (N == self.dim()
# concatenates the whole self shape; N is keyword-only) and whose dtype/device/
# layout come from ``other`` (the "feature meta"). The op never reads the
# payload values, so the value-range / nan-inf dimensions below only exercise
# input construction; the output is always an exact zero fill.
#
# Coverage:
#   * .default and .out overloads over N in [0, self.dim()] with 0-D/1-D/4-D
#     shapes on both sides, the same-object (self is other) path and zero-sized
#     dims;
#   * shape levels: pairs built from tu.selected_shapes() (quick/all
#     selected by --quick), with the output element count bounded so the
#     zero allocation stays cheap;
#   * value ranges: tu.selected_ranges() over representative shape pairs and
#     storage dtypes (the output is identical for every range);
#   * dtype contract: the output dtype follows ``other`` even when self and
#     other disagree;
#   * nan/inf payloads are ignored (the op only reads shapes and options);
#   * negative: N < 0, non-tensor arguments and a wrong-dtype .out tensor all
#     raise.
#
# No broadcast/backward dimensions apply: the operator never computes on the
# input values (there is nothing to broadcast or differentiate).

# N ranges over 0, interior batch sizes and N == self.dim() (full concat), and
# ranks 0-4 with 0-D/1-D tensors on both sides are covered, including zero-sized
# dims. The output element count stays small because the op only inspects
# metadata and allocates zeros.
_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES = [
    pytest.param((2, 3, 4, 5), (7, 8, 9), 0, id="N0"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 1, id="N1"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 3, id="N3"),
    pytest.param((2, 3, 4), (7, 8, 9), 3, id="N_self_rank"),
    pytest.param((2, 3), (4, 5, 6), 2, id="self_2d_other_3d"),
    pytest.param((3,), (4, 5), 1, id="self_1d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
    pytest.param((), (4, 5), 0, id="self_0d"),
    pytest.param((2, 3, 4), (5,), 3, id="other_1d"),
    pytest.param((2, 3), (), 2, id="other_0d"),
    pytest.param((0, 3), (4, 5), 1, id="self_zero_dim"),
    pytest.param((2, 3), (0, 5), 1, id="other_zero_dim"),
]

# The op performs no arithmetic: it only reads shapes/options and allocates a
# zero-filled tensor, so every storage dtype family the runtime supports is
# exercised.
_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

# Representative dtype subset for the shape-level sweep: a float pair, an int
# and bool. The full dtype matrix already runs over the main cases.
_SHAPE_LEVEL_DTYPES = utils.FLOAT_DTYPES + [torch.int32, torch.bool]

# Representative dtypes for the value-range sweep (the op ignores values).
_VALUE_RANGE_DTYPES = [torch.float32, torch.bfloat16, torch.int32, torch.bool]

# Pairs pinning down the "output options follow other" contract: the output
# dtype must be other.dtype even when self and other disagree.
_NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES = [
    pytest.param(torch.int16, torch.float16, id="self_int16_other_f16"),
    pytest.param(torch.float32, torch.bool, id="self_f32_other_bool"),
    pytest.param(torch.bool, torch.int32, id="self_bool_other_int32"),
    pytest.param(torch.float16, torch.float32, id="self_f16_other_f32"),
]

# Representative shape pairs for the value-range sweep.
_VALUE_RANGE_CASES = [
    pytest.param((2, 3, 4), (5, 6), 1, id="self_3d_other_2d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
]

# The op allocates a zero tensor of the concatenated shape; keep that
# allocation small so the shape-level sweep stays fast at every --quick.
_MAX_OUTPUT_ELEMENTS = 1_000_000


def _shape_level_cases():
    """Build (self_shape, other_shape, N) triples from tu.selected_shapes().

    Every level shape appears in the self position (N = 0 and, when the rank
    allows, N = 1 or N = 2) and in the other position; N never exceeds
    self.dim() and the output element count is bounded by
    ``_MAX_OUTPUT_ELEMENTS``.
    """
    cases = []
    for shape in tu.selected_shapes():
        cases.append((shape, (4, 5), 0))
        if len(shape) >= 1:
            cases.append((shape, (4, 5), 1))
        if len(shape) >= 2:
            cases.append((shape, (2,), 2))
        cases.append(((2,), shape, 1))
    return [
        (self_shape, other_shape, n)
        for self_shape, other_shape, n in cases
        if math.prod(self_shape[:n] + other_shape) <= _MAX_OUTPUT_ELEMENTS
    ]


def _make_value_tensor(dtype, shape, value_range, device):
    """Device-explicit twin of tu.make_input: same logic but on an explicit device."""
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
    if low == high:
        return torch.full(shape, low, device=device, dtype=dtype)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=device, low=low, high=high
    )


def _nan_inf_tensor(shape, dtype, device):
    """Build ``shape`` filled with nan/inf/-inf payloads (values the op
    ignores; the layout is plain and contiguous)."""
    t = torch.zeros(shape, dtype=dtype, device=device)
    n = t.numel()
    if n > 0:
        vals = torch.tensor(
            [float("nan"), float("inf"), float("-inf")], dtype=dtype, device=device
        )
        t = vals[torch.arange(n, device=device) % 3].reshape(shape)
    return t


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The .default and .out
    # overloads are resolved through their public operator names
    # "_new_zeros_with_same_feature_meta" and
    # "_new_zeros_with_same_feature_meta.out"; the direct flag_gems attribute
    # stays None until the op is registered, in which case resolve_gems_op falls
    # back to the override or raises LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_new_zeros_with_same_feature_meta",
        getattr(flag_gems, "_new_zeros_with_same_feature_meta", None),
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_new_zeros_with_same_feature_meta.out",
        getattr(flag_gems, "_new_zeros_with_same_feature_meta_out", None),
    )


def _resolve_gems_op_or_none():
    """Like _resolve_gems_op, but None while no candidate is registered yet."""
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _resolve_gems_op_out_or_none():
    try:
        return _resolve_gems_op_out()
    except LookupError:
        return None


def _assert_zero_output(res_out, ref_out, self_t, other_t, self_num_batch_dims):
    expected_shape = self_t.shape[:self_num_batch_dims] + other_t.shape
    assert isinstance(ref_out, torch.Tensor)
    assert ref_out.shape == expected_shape
    assert ref_out.is_contiguous()
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == other_t.dtype
    assert res_out.is_contiguous()
    assert res_out.storage_offset() == 0
    # A fresh allocation, not an alias of either input.
    assert res_out is not self_t and res_out is not other_t
    # The reference is all zeros, so exact equality also proves the candidate
    # produced a zero-filled tensor of the right shape and dtype.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    # The [-1, 1] range covers negative and positive values in every dtype.
    self_t = _make_value_tensor(dtype, self_shape, ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(dtype, other_shape, ["-1", "1"], flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = _resolve_gems_op()(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, self_num_batch_dims)


@pytest.mark._new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_out(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = _make_value_tensor(dtype, self_shape, ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(dtype, other_shape, ["-1", "1"], flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    # Pre-sized out tensors with non-zero garbage values: the .out variant must
    # overwrite them in place with zeros and return the same object.
    expected_shape = self_shape[:self_num_batch_dims] + other_shape
    res_out = torch.full(expected_shape, 7, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(expected_shape, 7, dtype=dtype, device=ref_other.device)

    ref_ret = torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims, out=ref_out
    )
    res_ret = _resolve_gems_op_out()(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims, out=res_out
    )

    # The .out variant must write into and return the out tensor itself.
    assert ref_ret is ref_out
    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, self_num_batch_dims)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _shape_level_cases()
)
@pytest.mark.parametrize("dtype", _SHAPE_LEVEL_DTYPES)
def test__new_zeros_with_same_feature_meta_shapes(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = _make_value_tensor(dtype, self_shape, ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(dtype, other_shape, ["-1", "1"], flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = _resolve_gems_op()(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, self_num_batch_dims)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _VALUE_RANGE_CASES
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test__new_zeros_with_same_feature_meta_value_ranges(
    self_shape, other_shape, self_num_batch_dims, value_range, dtype
):
    # The values sweep the full spec range set (positive, negative, extreme and
    # degenerate); the zero output never changes because the op reads only
    # shapes and options.
    self_t = _make_value_tensor(dtype, self_shape, value_range, flag_gems.device)
    other_t = _make_value_tensor(dtype, other_shape, value_range, flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = _resolve_gems_op()(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, self_num_batch_dims)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_dtype, other_dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES
)
def test__new_zeros_with_same_feature_meta_other_dtype_wins(self_dtype, other_dtype):
    self_t = _make_value_tensor(self_dtype, (2, 3, 4), ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(other_dtype, (7, 8), ["-1", "1"], flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = _resolve_gems_op()(self_t, other_t, self_num_batch_dims=1)

    assert ref_out.dtype == other_dtype
    _assert_zero_output(res_out, ref_out, self_t, other_t, 1)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_same_tensor(dtype):
    # self is other: the general shape formula still holds and the result is a
    # fresh zero allocation, not an alias of the shared input.
    self_t = _make_value_tensor(dtype, (2, 3, 4), ["-1", "1"], flag_gems.device)
    other_t = self_t
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = _resolve_gems_op()(self_t, other_t, self_num_batch_dims=1)

    assert ref_out.shape == (2, 2, 3, 4)
    _assert_zero_output(res_out, ref_out, self_t, other_t, 1)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__new_zeros_with_same_feature_meta_nan_inf_values(shape, dtype):
    # nan/inf/-inf are ordinary payloads that the allocation helper ignores;
    # the output is still an exact zero fill.
    self_t = _nan_inf_tensor(shape, dtype, flag_gems.device)
    other_t = _nan_inf_tensor((4, 5), dtype, flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=0
    )
    res_out = _resolve_gems_op()(self_t, other_t, self_num_batch_dims=0)

    _assert_zero_output(res_out, ref_out, self_t, other_t, 0)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__new_zeros_with_same_feature_meta_negative_batch_dims_raises(dtype):
    # self_num_batch_dims is the count of batch dims taken from self; a
    # negative count is invalid and the reference raises. A candidate must fail
    # loudly instead of silently slicing with a negative index.
    self_t = _make_value_tensor(dtype, (2, 3), ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(dtype, (4, 5), ["-1", "1"], flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=-1
        )
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(self_t, other_t, self_num_batch_dims=-1)


@pytest.mark._new_zeros_with_same_feature_meta_out
def test__new_zeros_with_same_feature_meta_out_wrong_dtype_raises():
    # The .out tensor must match the output options (which follow other); a
    # wrong-dtype out tensor is rejected by the reference and a candidate must
    # reject it too.
    self_t = _make_value_tensor(torch.float32, (2, 3), ["-1", "1"], flag_gems.device)
    other_t = _make_value_tensor(torch.float32, (4, 5), ["-1", "1"], flag_gems.device)
    out_t = torch.full((2, 4, 5), 1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta.out(
            self_t, other_t, self_num_batch_dims=1, out=out_t
        )
    gems_op = _resolve_gems_op_out_or_none()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(self_t, other_t, self_num_batch_dims=1, out=out_t)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_arg,other_arg",
    [
        pytest.param((1, 2), None, id="tuple_self"),
        pytest.param(1, None, id="int_self"),
        pytest.param(3.14, None, id="float_self"),
        pytest.param(None, 1, id="none_self"),
        pytest.param(None, None, id="none_both"),
    ],
)
def test__new_zeros_with_same_feature_meta_rejects_non_tensor(self_arg, other_arg):
    # The aten schema requires two Tensors; Python scalars/None hit the invalid
    # argument-combination path and raise. A candidate must fail too rather than
    # silently return a bogus allocation.
    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta(self_arg, other_arg)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(self_arg, other_arg)
