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

# ``_has_same_storage_numel`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._has_same_storage_numel`` and
# ``-m _has_same_storage_numel`` both work.
setattr(
    pytest.mark,
    "_has_same_storage_numel",
    MarkDecorator(
        Mark("_has_same_storage_numel", (), {}, _ispytest=True), _ispytest=True
    ),
)

# aten::_has_same_storage_numel(Tensor self, Tensor other) -> bool compares the
# *storage* element counts of the two tensors
# (self.storage().numel() == other.storage().numel()), not their logical numel.
# Views keep the full storage of their base, so a (4, 4) row slice still has a
# 16-element storage while an expanded (4, 4) tensor built from a (4, 1) base
# only has 4. The storage dtype does not affect the element count, so two
# tensors of different dtypes with the same storage length compare equal.
#
# It is a pure metadata query: the payload values (including nan/inf), the
# strides and the requires_grad flag never influence the result; the operator
# allocates nothing and returns a Python bool.
#
# Regular-operator-spec adaptation notes:
# - Broadcast: N/A -- the operator compares two independent storages; there is
#   nothing to broadcast against. Layout pairs (view/expand/transpose) are
#   covered instead because they are the meaningful "shape mismatch" dimension.
# - Backward: N/A -- the output is a plain bool with no autograd support, so
#   there is no gradient to compare.
# - Value ranges: the query never reads the element values, so the
#   tu.selected_ranges() grid verifies the same deterministic answer for every
#   storage range (positive, negative, extreme, degenerate).
# - nan/inf: covered by a dedicated case; non-finite payloads are ignored.
# - Negative: non-tensor / missing arguments are rejected at binding time.
#
# Shape coverage follows the regular-operator-spec level selection (quick/all
# via the pytest ``--quick`` flag): tu.selected_shapes() (0-D through 5-D).

# ---------------------------------------------------------------------------
# Dtype coverage -- probe, never guess
# ---------------------------------------------------------------------------
# The comparison ignores the storage values and dtype, so every storage dtype
# the runtime can allocate must be accepted. The spec's 9 required dtypes
# (int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/bf16/fp16/int32/int64) are probed with
# tu.supported_dtypes(); the wider float/int/bool families are added where the
# probe reports support.
_EXTRA_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.BOOL_TYPES
)
_CANDIDATE_DTYPES = list(dict.fromkeys(list(tu.REQUIRED_DTYPES) + _EXTRA_DTYPES))


def _dtype_probe(op_name, dtype):
    """Call the aten op on two tiny tensors of ``dtype``; any error = unsupported."""
    try:
        lhs = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
        rhs = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
        getattr(torch.ops.aten, op_name)(lhs, rhs)
        return True
    except Exception:
        return False


# If the probe yields nothing, keep the full candidate list rather than a
# float32-only fallback, so a failed/absent probe never silently drops the
# spec-required int8/uint8/fp8 dtypes.
_HAS_SAME_STORAGE_NUMEL_DTYPES = tu.supported_dtypes(
    "_has_same_storage_numel",
    candidates=_CANDIDATE_DTYPES,
    probe=_dtype_probe,
) or list(_CANDIDATE_DTYPES)

# Floating storage families used for the nan/inf payload case.
_FLOAT_STORAGE_DTYPES = [
    d for d in _HAS_SAME_STORAGE_NUMEL_DTYPES if d.is_floating_point
]

# ---------------------------------------------------------------------------
# Layout cases -- the semantic core of the operator
# ---------------------------------------------------------------------------
# Each case is a pair of storage-layout specs. ``plain`` tensors carry a storage
# whose element count equals their logical numel; the other kinds deliberately
# decouple logical shape from storage length so a candidate that reads
# ``tensor.numel()`` instead of the storage length is caught.
_HAS_SAME_STORAGE_NUMEL_CASES = [
    pytest.param(("plain", (4, 4)), ("plain", (4, 4)), id="same_shape_true"),
    pytest.param(("plain", (4, 4)), ("plain", (16,)), id="reshaped_same_storage_true"),
    pytest.param(("plain", (4, 4)), ("plain", (8,)), id="different_numel_false"),
    pytest.param(
        ("plain", (4, 4)), ("transposed", (4, 4)), id="transposed_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("row_view", (4, 4)), id="row_slice_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("narrowed", (4, 4)), id="narrowed_same_storage_true"
    ),
    pytest.param(
        ("plain", (4, 4)), ("expanded", (4, 4)), id="plain_larger_storage_false"
    ),
    pytest.param(
        ("expanded", (4, 4)), ("plain", (4,)), id="expanded_base_matches_false"
    ),
    pytest.param(
        ("row_view", (4, 4)), ("plain", (4,)), id="row_slice_larger_storage_false"
    ),
    pytest.param(
        ("narrowed", (16,)), ("plain", (4,)), id="narrowed_larger_storage_false"
    ),
    pytest.param(("plain", ()), ("plain", (1,)), id="scalar_vs_single_true"),
    pytest.param(("plain", (0,)), ("plain", (0, 5)), id="empty_same_storage_true"),
    pytest.param(("plain", (0,)), ("plain", (3,)), id="empty_vs_nonempty_false"),
]

# Different storage dtypes still share an element count, so the answer is
# ``True`` whenever the storage lengths agree.
_CROSS_DTYPE_CASES = [
    pytest.param(torch.float32, torch.int64, id="fp32_vs_int64"),
    pytest.param(torch.float16, torch.float32, id="fp16_vs_fp32"),
    pytest.param(torch.int8, torch.uint8, id="int8_vs_uint8"),
    pytest.param(torch.bfloat16, torch.float8_e4m3fn, id="bf16_vs_fp8"),
    pytest.param(torch.bool, torch.int32, id="bool_vs_int32"),
]

# Non-tensor arguments: the aten schema requires (Tensor, Tensor); Python
# scalars / None / sequences hit the invalid argument-combination path.
_INVALID_ARG_CASES = [
    pytest.param((1, 2), None, id="tuple_self"),
    pytest.param(1, None, id="int_self"),
    pytest.param(3.14, None, id="float_self"),
    pytest.param(None, 1, id="none_self"),
    pytest.param(None, None, id="none_both"),
    pytest.param("abc", "abc", id="str_both"),
]


def _make_tensor(spec, dtype, device):
    """Build a tensor with the requested storage-layout spec on ``device``."""
    kind, shape = spec
    if kind == "plain":
        return torch.zeros(shape, dtype=dtype, device=device)
    if kind == "transposed":
        return torch.zeros((shape[1], shape[0]), dtype=dtype, device=device).t()
    if kind == "row_view":
        return torch.zeros(shape, dtype=dtype, device=device)[0]
    if kind == "expanded":
        base = torch.zeros((shape[0], 1), dtype=dtype, device=device)
        return base.expand(shape)
    if kind == "narrowed":
        base = torch.zeros(shape, dtype=dtype, device=device)
        return base.narrow(0, shape[0] // 4, max(shape[0] // 2, 1))
    raise ValueError(f"Unknown tensor spec kind: {kind!r}")


def _make_value_tensor(dtype, shape, value_range, device):
    """Device-aware value-range tensor builder.

    Mirrors ``tu.make_input`` but additionally (a) clamps integer bounds to the
    dtype's representable range -- an unsigned dtype over ``["-1", "0"]`` would
    otherwise collapse to the invalid interval ``[0, 0]`` rejected by
    ``torch.testing.make_tensor`` -- and (b) allows an explicit device so the
    CPU reference keeps the same contiguous storage layout.
    """
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()

    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if not (dtype.is_floating_point or dtype.is_complex):
        info = torch.iinfo(dtype)
        low, high = max(int(low), info.min), min(int(high), info.max)

    if low == high:
        return torch.full(shape, low, dtype=dtype, device=device)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=device, low=low, high=high
    )


def _nan_inf_tensor(shape, dtype, device):
    """Build ``shape`` filled with a nan/inf/-inf payload the query ignores."""
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
    # override installed by KernelGen for this run wins. ``flag_gems.
    # _has_same_storage_numel`` may not be registered yet, so getattr supplies a
    # safe default and resolve_gems_op falls back to the package namespace
    # before raising LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_has_same_storage_numel",
        getattr(flag_gems, "_has_same_storage_numel", None),
    )


def _resolve_gems_op_or_none():
    """Like ``_resolve_gems_op`` but returns None while no candidate exists."""
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _assert_result(res_out, ref_out):
    # The op returns a plain Python bool; a candidate may equivalently return a
    # 0-dim bool tensor. The comparison is exact (no tolerance involved).
    assert isinstance(ref_out, bool)
    res_t = torch.as_tensor(res_out).detach().cpu().reshape(())
    assert res_t.dtype == torch.bool
    utils.gems_assert_equal(res_t, torch.tensor(ref_out, dtype=torch.bool))


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_spec,other_spec", _HAS_SAME_STORAGE_NUMEL_CASES)
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_layouts(self_spec, other_spec, dtype):
    self_t = _make_tensor(self_spec, dtype, flag_gems.device)
    other_t = _make_tensor(other_spec, dtype, flag_gems.device)

    # Build the reference from the same storage-layout spec on the reference
    # device: moving a view to CPU would compact its storage and change the
    # answer, so both sides must be constructed with identical layouts.
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = _make_tensor(self_spec, dtype, ref_device)
    ref_other = _make_tensor(other_spec, dtype, ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_dtype,other_dtype", _CROSS_DTYPE_CASES)
def test__has_same_storage_numel_cross_dtype(self_dtype, other_dtype):
    # The storage *dtype* is irrelevant to the element count: identical storage
    # lengths must compare equal even across dtype families.
    self_t = torch.zeros((4, 4), dtype=self_dtype, device=flag_gems.device)
    other_t = torch.zeros((16,), dtype=other_dtype, device=flag_gems.device)
    ref_self = self_t.to("cpu") if utils.TO_CPU else self_t
    ref_other = other_t.to("cpu") if utils.TO_CPU else other_t

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_shapes(shape, dtype):
    # Shape-level coverage from the shared selector: plain tensors of the same
    # logical shape share a storage of the same numel, so the answer is True at
    # every level (0-D scalar through 5-D).
    self_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_value_ranges(shape, value_range, dtype):
    # The values sweep the full spec range set (positive, negative, extreme and
    # degenerate); the reported comparison never changes because the query reads
    # only storage metadata. Same-shape inputs always answer True.
    self_t = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    other_t = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _FLOAT_STORAGE_DTYPES)
def test__has_same_storage_numel_nan_inf(shape, dtype):
    # nan/inf/-inf are ordinary payloads that the metadata query must ignore;
    # the answer is still the storage-numel comparison of the two tensors.
    self_t = _nan_inf_tensor(shape, dtype, flag_gems.device)
    other_t = _nan_inf_tensor(shape, dtype, flag_gems.device)
    ref_self = utils.to_reference(self_t)
    ref_other = utils.to_reference(other_t)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel_ignores_autograd():
    # The query has no autograd support: a requires_grad input must neither
    # change the answer nor produce a differentiable output.
    self_t = torch.zeros((4, 4), device=flag_gems.device).requires_grad_()
    other_t = torch.zeros((4, 4), device=flag_gems.device).requires_grad_()
    ref_self = self_t.detach()
    ref_other = other_t.detach()

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert not isinstance(res_out, torch.Tensor) or not res_out.requires_grad


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_arg,other_arg", _INVALID_ARG_CASES)
def test__has_same_storage_numel_rejects_non_tensor(self_arg, other_arg):
    # The aten schema requires two Tensors; Python scalars/None hit the invalid
    # argument-combination path and raise. A candidate must fail too rather than
    # silently return a bogus comparison.
    with pytest.raises(RuntimeError):
        torch.ops.aten._has_same_storage_numel(self_arg, other_arg)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        # The reference raises RuntimeError at the dispatcher level; a
        # plain-Python candidate naturally raises AttributeError (or a
        # TypeError/ValueError) for the same inputs, which is equally
        # acceptable.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(self_arg, other_arg)


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel_rejects_missing_argument():
    # Wrong arity must be rejected as well.
    inp = torch.zeros((4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._has_same_storage_numel(inp)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(inp)
