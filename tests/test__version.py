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

# ``_version`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._version`` and ``-m _version`` both
# work.
setattr(
    pytest.mark,
    "_version",
    MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_version(Tensor self) -> int reports the version counter of a tensor
# (the number of in-place mutations applied to it so far, shared between a base
# tensor and all of its views). It is a pure O(1) metadata query whose result
# never depends on the shape, layout, or payload values of the storage, so the
# value-range / nan-inf sweeps below exercise every supported dtype family while
# always expecting the same answer.
#
# Coverage:
#   * fresh tensors start at version 0 (0-D to 5-D, all float dtypes);
#   * every storage dtype family the runtime supports is accepted;
#   * shape levels: tu.selected_shapes() (quick/core/all via TEST_LEVEL);
#   * value ranges: tu.selected_ranges() over representative shapes -- the
#     answer is data-independent, so it is 0 for every range;
#   * nan/inf payloads are ignored by the metadata query;
#   * every in-place mutation bumps the counter by exactly one;
#   * a read-only call neither bumps the counter nor touches the data;
#   * views share the version counter with their base (including mutations
#     applied through the view);
#   * negative: non-tensor arguments are rejected.
#
# No broadcast/backward dimensions apply: the operator takes a single Tensor,
# returns a plain int, and is not differentiable.

# Deliberately small because the op is O(1), but covers every rank from 0-D to
# 5-D so the candidate must accept arbitrary tensors.
_VERSION_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (3, 4), (8, 16, 4), (2, 3, 4, 5), (4, 7, 5, 3, 2)]
)

# The version counter lives on the TensorImpl, so the storage dtype is
# irrelevant; still, cover every dtype family the runtime supports.
_VERSION_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Representative shapes for the value-range sweep: 0-D scalar, 1-D, 2-D and 3-D
# so every level keeps the cross product bounded while still covering all ranks
# the op accepts.
_VALUE_RANGE_SHAPES = (
    [(2, 19, 7)] if tu.LEVEL == "quick" else [(), (256,), (1024, 1024), (7, 13, 29)]
)

# Non-tensor arguments: the aten schema requires one Tensor. None is excluded
# because the dispatcher silently returns a default-constructed 0 for it; the
# remaining Python scalars/sequences hit the invalid argument-combination path
# and raise RuntimeError on the reference.
_INVALID_ARG_CASES = [
    pytest.param(1, id="int"),
    pytest.param(3.14, id="float"),
    pytest.param("string", id="str"),
    pytest.param([1, 2], id="list"),
]


def _make_input(shape, dtype):
    # _version never touches the tensor data, so zeros are sufficient.
    return torch.zeros(shape, dtype=dtype, device=flag_gems.device)


def _make_value_tensor(dtype, shape, value_range, device):
    """Device-explicit twin of tu.make_input (which hardcodes its own device)."""
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
    # override installed by KernelGen for this run wins. ``flag_gems._version``
    # is the package version module (package metadata), not the operator
    # callable, so the default is left as None; resolution order stays:
    # (1) the process-local override injected by KernelGen, (2) a registered
    # ``flag_gems._version`` operator callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_version", None)


def _resolve_gems_op_or_none():
    """Like _resolve_gems_op, but None while no candidate is registered yet."""
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _as_int(value):
    # The reference returns a plain Python int; a candidate may equivalently
    # return a 0-dim integral tensor. Normalize both before comparing.
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return value


def _assert_result(res_out, ref_out):
    # _version returns a plain Python int holding the version counter, so exact
    # equality is required and no tolerance is involved.
    res_int = _as_int(res_out)
    ref_int = _as_int(ref_out)
    assert type(res_int) is int
    assert type(ref_int) is int
    utils.gems_assert_equal(res_int, ref_int)


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_fresh(shape, dtype):
    # A freshly created tensor starts at version 0.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == 0


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_dtype_coverage(dtype):
    # The version counter ignores the storage dtype; every dtype family must be
    # accepted and reported as 0 for a fresh tensor.
    inp = _make_input((3, 4), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_shape_levels(shape, dtype):
    # Shape-level coverage from the shared selector: fresh tensors report
    # version 0 at every level (0-D scalar through 8-D) and dtype family.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", _VALUE_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_value_ranges(shape, value_range, dtype):
    # The values sweep the full spec range set (positive, negative, extreme and
    # degenerate); the reported version never changes because the query reads
    # only the TensorImpl version counter, never the payload. The absolute value
    # is not fixed at 0 here: torch.testing.make_tensor fills float tensors
    # in-place and therefore bumps the counter to 1, so the candidate must only
    # agree with the reference (which observes the same construction path).
    inp = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _make_value_tensor(dtype, shape, value_range, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__version_nan_inf_values(shape, dtype):
    # nan/inf/-inf are ordinary payloads that the metadata query must ignore;
    # a fresh tensor holding them still reports version 0.
    inp = _nan_inf_tensor(shape, dtype, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _nan_inf_tensor(shape, dtype, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("bumps", [1, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_after_inplace(shape, bumps, dtype):
    # Every in-place mutation increments the version counter by one; the op
    # must report the exact number of mutations applied to the tensor. The
    # reference input may be the same tensor (CPU reference disabled) or an
    # independent copy (TO_CPU), so bump it exactly ``bumps`` times either way.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    for _ in range(bumps):
        torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        for _ in range(bumps):
            torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == bumps


@pytest.mark._version
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_readonly(dtype):
    # _version is a read-only query: it must neither bump the version counter
    # nor modify the tensor data.
    inp = _make_input((8, 16), dtype)
    ref_inp = utils.to_reference(inp)
    data_before = inp.clone()
    version_before = torch.ops.aten._version(ref_inp)

    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, version_before)
    assert torch.ops.aten._version(inp) == version_before
    assert torch.equal(inp, data_before)


@pytest.mark._version
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_view(dtype):
    # Views share the version counter with their base tensor, so the op must
    # report the same value for a view as for the base.
    inp = _make_input((4, 6), dtype)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    ref_base = torch.ops.aten._version(ref_inp)
    ref_out = torch.ops.aten._version(ref_view)
    res_out = _resolve_gems_op()(view)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == ref_base


@pytest.mark._version
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_view_inplace(dtype):
    # In-place mutation through a view bumps the shared version counter, so the
    # base tensor must report the same bumped value as the view.
    inp = _make_input((4, 6), dtype)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    torch.ops.aten.add_.Tensor(view, 1)
    if ref_view is not view:
        torch.ops.aten.add_.Tensor(ref_view, 1)

    ref_base = torch.ops.aten._version(ref_inp)
    ref_out = torch.ops.aten._version(ref_view)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert _as_int(res_out) == ref_base


@pytest.mark._version
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test__version_rejects_non_tensor(bad_arg):
    # The aten schema requires a single Tensor; Python scalars/sequences hit the
    # invalid argument-combination path and raise. A candidate must fail too
    # rather than silently return a bogus version.
    with pytest.raises(RuntimeError):
        torch.ops.aten._version(bad_arg)
    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        # A candidate must fail loudly instead of silently returning a bogus
        # version. The reference raises RuntimeError at the dispatcher level; a
        # plain-Python candidate naturally raises AttributeError (or a
        # TypeError/ValueError) for the same inputs, which is equally
        # acceptable.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(bad_arg)
