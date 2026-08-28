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
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

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
# is ``self.shape[:self_num_batch_dims] + other.shape`` and whose dtype/device/
# layout come from ``other`` (the "feature meta"). ``self_num_batch_dims`` is
# keyword-only. Every case below is a distinct parametrized workload; N ranges
# over 0, interior batch sizes and N == self.dim() (full concat), and ranks
# 0-4 with 0-D/1-D tensors on both sides are covered. The output element count
# stays small because the op only inspects metadata and allocates zeros.
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
]

# The op performs no arithmetic: it only reads shapes/options and allocates a
# zero-filled tensor, so every storage dtype family the runtime supports is
# exercised.
_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

# Pairs pinning down the "output options follow other" contract: the output
# dtype must be other.dtype even when self and other disagree.
_NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES = [
    pytest.param(torch.int16, torch.float16, id="self_int16_other_f16"),
    pytest.param(torch.float32, torch.bool, id="self_f32_other_bool"),
    pytest.param(torch.bool, torch.int32, id="self_bool_other_int32"),
    pytest.param(torch.float16, torch.float32, id="self_f16_other_f32"),
]


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


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
    self_t = _make_input(self_shape, dtype)
    other_t = _make_input(other_shape, dtype)
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
    self_t = _make_input(self_shape, dtype)
    other_t = _make_input(other_shape, dtype)
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
    "self_dtype, other_dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES
)
def test__new_zeros_with_same_feature_meta_other_dtype_wins(self_dtype, other_dtype):
    self_t = _make_input((2, 3, 4), self_dtype)
    other_t = _make_input((7, 8), other_dtype)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = _resolve_gems_op()(self_t, other_t, self_num_batch_dims=1)

    assert ref_out.dtype == other_dtype
    _assert_zero_output(res_out, ref_out, self_t, other_t, 1)
