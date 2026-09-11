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

# ``_version`` starts with an underscore and ``pytest.mark`` refuses to
# generate a marker through attribute access for such names, so register it
# directly on the MarkGenerator to keep both ``@pytest.mark._version`` and
# ``-m _version`` working.
try:
    pytest.mark._version
except AttributeError:
    setattr(
        pytest.mark,
        "_version",
        MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_version(Tensor self) -> int returns the version counter of a tensor:
# the number of in-place mutations applied to its TensorImpl so far. The
# counter is shared between a tensor and its aliases (views, detach), while
# unrelated tensors count independently.
#
# It is a pure O(1) metadata query: the result never depends on the shape,
# layout, storage dtype or payload values, so the sweeps below exercise all of
# those while always comparing against the reference produced by
# torch.ops.aten._version.
#
# Coverage map (regular-operator spec):
#   * dtypes: the 9 required dtypes probed with tu.supported_dtypes (int8 /
#     uint8 / float8_e4m3fn / float8_e5m2 / fp32 / bf16 / fp16 / int32 /
#     int64) plus the shared float / int / bool / complex families where the
#     active backend supports them (the counter ignores the storage dtype);
#   * shapes: tu.selected_shapes() -- the shared 7 levels (0-D scalar to 5-D),
#     driven by the pytest --quick flag;
#   * value ranges: the shared tu.selected_ranges() grid over every shape and
#     the 9 required dtypes (5 ranges x 7 shapes x 9 dtypes);
#   * nan / inf / -inf payloads are ignored by the metadata query;
#   * in-place mutations bump the counter by exactly one each (1/2/3/5 bumps);
#   * read-only semantics: the query neither bumps the counter nor writes;
#   * alias semantics: views and detach() share the counter with the base, and
#     a mutation applied through a view bumps the base counter;
#   * independence: unrelated tensors keep separate counters;
#   * negative cases: non-tensor arguments, a missing argument and an extra
#     argument are rejected.
#
# No broadcast dimension applies (the operator is unary) and no backward
# dimension applies (it returns a plain int and has no autograd formula).

# Small ranks used by the mutation / alias workloads (kept lightweight because
# every bump is an element-wise op over the whole tensor).
_VERSION_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (3, 4), (8, 16, 4), (2, 3, 4, 5), (4, 7, 5, 3, 2)]
)

_FP8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
}


def _dedup(dtypes):
    seen = set()
    ordered = []
    for dtype in dtypes:
        if dtype not in seen:
            seen.add(dtype)
            ordered.append(dtype)
    return ordered


def _supported_or(candidates, fallback):
    supported = tu.supported_dtypes("_version", candidates=candidates)
    return list(supported) if supported else list(fallback)


# The five-range / seven-shape grid runs on the required dtype set, probed on
# the active device so unsupported storages are skipped instead of failing.
_GRID_DTYPES = _supported_or(tu.REQUIRED_DTYPES, tu.REQUIRED_DTYPES)

# Additional dtype families from the shared selector, probed the same way so a
# backend without fp64 / fp8 / complex storage degrades cleanly.
_EXTRA_DTYPE_CANDIDATES = _dedup(
    list(utils.ALL_FLOAT_DTYPES)
    + list(utils.ALL_INT_DTYPES)
    + list(utils.BOOL_TYPES)
    + list(utils.COMPLEX_DTYPES)
)
_VERSION_DTYPES = _dedup(
    _GRID_DTYPES + _supported_or(_EXTRA_DTYPE_CANDIDATES, _EXTRA_DTYPE_CANDIDATES)
)

# nan / inf / -inf need a dtype with an inf value (fp8 has none).
_FLOAT_VALUE_DTYPES = [dtype for dtype in _VERSION_DTYPES if dtype.is_floating_point]

# Dtypes that accept an in-place ``add_`` bump (bool and fp8 reject it on the
# CUDA backend), used by the mutation / alias workloads.
_MUTABLE_DTYPES = [
    dtype
    for dtype in _VERSION_DTYPES
    if dtype != torch.bool and dtype not in _FP8_DTYPES
]

# Non-tensor arguments. ``None`` is deliberately excluded because the
# dispatcher silently returns a default-constructed 0 for it.
_INVALID_ARG_CASES = [
    pytest.param(1, id="int"),
    pytest.param(3.14, id="float"),
    pytest.param("string", id="str"),
    pytest.param([1, 2], id="list"),
]


def _make_value_tensor(dtype, shape, value_range, device):
    """Value-range helper mirroring ``tu.make_input`` with an explicit device.

    The reference must be built by the *same* construction path on its own
    device: ``Tensor.to("cpu")`` resets the version counter, so a device copy
    would not be comparable. ``torch.testing.make_tensor`` produces the same
    counter on CPU and on the device for every dtype/range used below.
    """
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, _ = tu.dtype_bounds(dtype)
        if low < int(dtype_min):
            # Unsigned dtypes cannot represent the "-1" low symbol; snap it to
            # the representable minimum (the "min" symbol already resolves to
            # the dtype minimum through tu.resolve_bound).
            low = int(dtype_min)

    if low == high:
        return torch.full(shape, low, device=device, dtype=dtype)

    try:
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=device, low=low, high=high
        )
    except RuntimeError:
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype_min, dtype_max = int(dtype_min), int(dtype_max)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=device, low=dtype_min, high=dtype_max
        )


def _nan_inf_tensor(shape, dtype, device):
    """Build ``shape`` holding a nan / inf / -inf payload cycle.

    The values are ignored by the query, but the layout stays plain and
    contiguous.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf")], dtype=dtype, device=device
    )
    index = torch.arange(numel, device=device) % 3
    return values[index].reshape(shape)


def _default_gems_op():
    # ``flag_gems._version`` is the package version string (package metadata),
    # not an operator callable; treat any non-callable attribute as "no
    # default" so resolution falls through to the KernelGen override.
    candidate = getattr(flag_gems, "_version", None)
    return candidate if callable(candidate) else None


def _resolve_gems_op():
    # Resolved inside every test (never at import time) so the process-local
    # override injected by KernelGen for this run wins. The resolution order is
    # (1) the process-local override, (2) the direct FlagGems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_version", _default_gems_op())


def _resolve_gems_op_or_none():
    try:
        return _resolve_gems_op()
    except LookupError:
        return None


def _as_int(value):
    # The reference returns a plain Python int; a candidate may equivalently
    # return a 0-dim / single-element integral tensor. Normalize both.
    if isinstance(value, torch.Tensor):
        assert value.numel() == 1, "candidate returned a non-scalar tensor"
        return int(value.item())
    return value


def _assert_result(res_out, ref_out):
    # Exact equality: the op reports a mutation count, so no tolerance applies.
    res_int = _as_int(res_out)
    ref_int = _as_int(ref_out)
    assert isinstance(res_int, int) and not isinstance(res_int, bool)
    assert isinstance(ref_int, int) and not isinstance(ref_int, bool)
    utils.gems_assert_equal(res_int, ref_int)
    return res_int, ref_int


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_fresh(shape, dtype):
    # A freshly created tensor starts at version 0 at every shape level and for
    # every storage dtype the backend supports.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test__version_value_ranges(shape, value_range, dtype):
    # The shared value-range grid ([-1,1], [0,1], [-1,0], [0,max], [min,0]) over
    # the shared shape levels and the required dtypes: the payload never affects
    # the metadata query. Construction itself may bump the counter (make_tensor
    # fills floats in place), so both sides go through the identical
    # construction path and stay comparable.
    inp = _make_value_tensor(dtype, shape, value_range, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _make_value_tensor(dtype, shape, value_range, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test__version_nan_inf(shape, dtype):
    # nan / inf / -inf are ordinary payloads the metadata query must ignore; a
    # freshly built tensor holding them still reports version 0.
    inp = _nan_inf_tensor(shape, dtype, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _nan_inf_tensor(shape, dtype, ref_device)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("bumps", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_after_inplace(shape, bumps, dtype):
    # Every in-place mutation increments the counter by exactly one; the op must
    # report the exact number of bumps applied. The reference tensor is created
    # before the loop (version 0 on both devices) and bumped identically.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    for _ in range(bumps):
        torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        for _ in range(bumps):
            torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == bumps


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_readonly(dtype):
    # _version is a read-only query: it must neither bump the counter nor write
    # to the tensor.
    inp = torch.zeros((8, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    data_before = inp.clone()
    version_before = torch.ops.aten._version(ref_inp)

    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, version_before)
    assert torch.ops.aten._version(inp) == version_before
    assert torch.equal(inp, data_before)


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_view(dtype):
    # Views share the version counter with their base, so a view reports the
    # same value as the base tensor.
    inp = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    ref_base = torch.ops.aten._version(ref_inp)
    ref_out = torch.ops.aten._version(ref_view)
    res_out = _resolve_gems_op()(view)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == ref_base == 0


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_view_inplace(dtype):
    # An in-place mutation applied through a view bumps the shared counter, so
    # the base tensor must report the same bumped value as the view.
    inp = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    torch.ops.aten.add_.Tensor(view, 1)
    if ref_inp is not inp:
        torch.ops.aten.add_.Tensor(ref_view, 1)

    ref_out = torch.ops.aten._version(ref_view)
    res_out = _resolve_gems_op()(inp)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == 1


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_detach_shares_counter(dtype):
    # detach() keeps the same TensorImpl version counter as the source, so a
    # mutation of either side is visible through both.
    inp = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    detached = inp.detach()
    ref_detached = ref_inp.detach()

    torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_detached)
    res_out = _resolve_gems_op()(detached)

    _, ref_int = _assert_result(res_out, ref_out)
    assert ref_int == 1


@pytest.mark._version
@pytest.mark.parametrize("dtype", _MUTABLE_DTYPES)
def test__version_independent_counters(dtype):
    # Unrelated tensors have independent counters: bumping one must not affect
    # the version reported for the other.
    first = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    second = torch.zeros((4,), dtype=dtype, device=flag_gems.device)
    ref_first = utils.to_reference(first)
    ref_second = utils.to_reference(second)

    torch.ops.aten.add_.Tensor(first, 1)
    torch.ops.aten.add_.Tensor(first, 1)
    torch.ops.aten.add_.Tensor(second, 1)
    if ref_first is not first:
        torch.ops.aten.add_.Tensor(ref_first, 1)
        torch.ops.aten.add_.Tensor(ref_first, 1)
        torch.ops.aten.add_.Tensor(ref_second, 1)

    assert torch.ops.aten._version(ref_first) == 2
    assert torch.ops.aten._version(ref_second) == 1

    _assert_result(_resolve_gems_op()(first), 2)
    _assert_result(_resolve_gems_op()(second), 1)


@pytest.mark._version
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test__version_rejects_non_tensor(bad_arg):
    # The aten schema requires a single Tensor; Python scalars and sequences hit
    # the invalid argument-combination path and raise. A candidate must fail
    # loudly too instead of returning a bogus version.
    with pytest.raises(RuntimeError):
        torch.ops.aten._version(bad_arg)

    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        # The reference raises RuntimeError at the dispatcher level; a plain
        # Python candidate naturally raises AttributeError / TypeError /
        # ValueError for the same inputs, which is equally acceptable.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op(bad_arg)


@pytest.mark._version
def test__version_rejects_wrong_arity():
    # Missing and extra positional arguments are rejected by the schema.
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten._version()

    extra = torch.zeros(2, device=flag_gems.device)
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten._version(extra, 1)

    # The single Tensor argument may be passed by keyword.
    assert torch.ops.aten._version(self=extra) == 0

    gems_op = _resolve_gems_op_or_none()
    if gems_op is not None:
        # A candidate fails on a missing argument with whatever the runtime
        # raises for a wrong arity: the reference (packet / bound method) raises
        # RuntimeError, while a plain Python implementation raises TypeError.
        with pytest.raises((TypeError, RuntimeError)):
            gems_op()
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(extra, 1)
