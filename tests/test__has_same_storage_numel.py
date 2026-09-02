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
from _pytest.mark.structures import Mark, MarkDecorator

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
# *storage* element counts of the two tensors (self.storage().numel() ==
# other.storage().numel()), not their logical numel. Views keep the full
# storage of their base, so a (4, 4) row slice still has a 16-element storage
# while an expanded (4, 4) tensor built from a (4, 1) base only has 4. It is a
# pure metadata query: the payload values (including nan/inf) and the storage
# dtype never influence the result.
#
# Coverage:
#   * layout pairs: plain/transposed/row-slice/narrowed/expanded views, both
#     True and False outcomes, including cases where the logical shapes agree
#     but the storage sizes differ;
#   * shape levels: tu.selected_shapes() plain-vs-plain pairs (quick/all
#     selected by --quick), 0-D scalar through 8-D;
#   * value ranges: tu.selected_ranges() over representative shapes, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the answer is identical for all of them);
#   * nan/inf payloads are ignored by the metadata query;
#   * negative: non-tensor arguments are rejected.
#
# No broadcast/backward dimensions apply: the operator returns a plain bool and
# the comparison is between two independent storages (there is nothing to
# broadcast against or differentiate).

_HAS_SAME_STORAGE_NUMEL_CASES = [
    pytest.param(("plain", (4, 4)), ("plain", (4, 4)), id="same_shape"),
    pytest.param(("plain", (4, 4)), ("plain", (16,)), id="same_numel_reshaped"),
    pytest.param(("plain", (4, 4)), ("plain", (8,)), id="different_numel"),
    pytest.param(("plain", (4, 4)), ("transposed", (4, 4)), id="transposed_view"),
    pytest.param(("plain", (4, 4)), ("row_view", (4, 4)), id="row_slice_view"),
    pytest.param(("plain", (4, 4)), ("narrowed", (16,)), id="narrowed_same_storage"),
    pytest.param(
        ("plain", (4, 4)), ("expanded", (4, 4)), id="expanded_storage_smaller"
    ),
    pytest.param(("expanded", (4, 4)), ("plain", (4,)), id="expanded_storage_equal"),
    pytest.param(("row_view", (4, 4)), ("plain", (4,)), id="row_slice_storage_larger"),
    pytest.param(("narrowed", (16,)), ("plain", (4,)), id="narrowed_storage_larger"),
    pytest.param(("plain", ()), ("plain", (1,)), id="scalar_vs_single"),
]

# The op is a pure storage-metadata query: it never reads the tensor values, so
# every storage dtype family the runtime supports is exercised.
_HAS_SAME_STORAGE_NUMEL_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

# Representative layouts for the value-range sweep: 0-D scalar, 1-D, 2-D and 3-D
# so every level keeps the cross product bounded while still covering all ranks
# the op accepts.
_VALUE_RANGE_SHAPES = (
    [(2, 19, 7)] if tu.LEVEL == "quick" else [(), (256,), (1024, 1024), (7, 13, 29)]
)

# Non-tensor arguments: the aten schema requires (Tensor, Tensor), and a Python
# scalar/None/sequence hits the invalid argument-combination path.
_INVALID_ARG_CASES = [
    pytest.param((1, 2), None, id="tuple_self"),
    pytest.param(1, None, id="int_self"),
    pytest.param(3.14, None, id="float_self"),
    pytest.param(None, 1, id="none_self"),
    pytest.param(None, None, id="none_both"),
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
        return torch.zeros(shape, dtype=dtype, device=device).narrow(
            0, shape[0] // 4, shape[0] // 2
        )
    raise ValueError(f"Unknown tensor spec kind: {kind!r}")


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
    """Build ``shape`` filled with a nan/inf/-inf payload (values the query
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
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._has_same_storage_numel is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._has_same_storage_numel
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_has_same_storage_numel",
        getattr(flag_gems, "_has_same_storage_numel", None),
    )


def _resolve_gems_op_or_none():
    """Like _resolve_gems_op, but None while no candidate is registered yet."""
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _assert_result(res_out, ref_out):
    # The op returns a plain Python bool; a candidate may equivalently return a
    # 0-dim bool tensor. The comparison is exact (no tolerance involved).
    assert isinstance(ref_out, bool)
    assert isinstance(res_out, (bool, torch.Tensor))
    utils.gems_assert_equal(
        torch.tensor(res_out, device="cpu"), torch.tensor(ref_out, device="cpu")
    )


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_spec,other_spec", _HAS_SAME_STORAGE_NUMEL_CASES)
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel(self_spec, other_spec, dtype):
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
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_shapes(shape, dtype):
    # Shape-level coverage from the shared selector: plain tensors of the same
    # logical shape share a storage of the same numel, so the answer is True at
    # every level (0-D scalar through 8-D).
    self_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = torch.zeros(shape, dtype=dtype, device=ref_device)
    ref_other = torch.zeros(shape, dtype=dtype, device=ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", _VALUE_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel_value_ranges(shape, value_range, dtype):
    # The values sweep the full spec range set (positive, negative, extreme and
    # degenerate); the reported comparison never changes because the query reads
    # only storage metadata. Same-shape inputs always answer True.
    self_t = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    other_t = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = _make_value_tensor(dtype, shape, value_range, ref_device)
    ref_other = _make_value_tensor(dtype, shape, value_range, ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__has_same_storage_numel_nan_inf_values(shape, dtype):
    # nan/inf/-inf are ordinary payloads that the metadata query must ignore;
    # the answer is still the storage-numel comparison of the two tensors.
    self_t = _nan_inf_tensor(shape, dtype, flag_gems.device)
    other_t = _nan_inf_tensor(shape, dtype, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = _nan_inf_tensor(shape, dtype, ref_device)
    ref_other = _nan_inf_tensor(shape, dtype, ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    _assert_result(res_out, ref_out)
    assert bool(ref_out) is True


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
        # A candidate must fail loudly instead of silently returning a bogus
        # comparison. The reference raises RuntimeError at the dispatcher level;
        # a plain-Python candidate naturally raises AttributeError (or a
        # TypeError/ValueError) for the same inputs, which is equally
        # acceptable.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(self_arg, other_arg)
