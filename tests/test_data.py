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

# aten::data(Tensor self) -> Tensor is the deprecated ``Tensor.data`` accessor:
# it returns a new tensor that shares the input's storage (same data_ptr, shape,
# stride and storage_offset) and is detached from autograd (requires_grad=False,
# is_leaf=True, grad_fn=None), i.e. it behaves like detach() + view. No
# arithmetic happens at the call, so every storage dtype is supported and the
# observed values round-trip bit-for-bit.
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * dtype grid: the nine required spec dtypes (int8, uint8, fp8-e4m3fn,
#     fp8-e5m2, fp32, bf16, fp16, int32, int64), plus fp64/int16/complex64/bool
#     when the runtime accepts them, probed with tu.supported_dtypes so a
#     backend that lacks e.g. fp8 drops that dtype instead of failing;
#   * shape levels: tu.selected_shapes() (ranks 0-5, selected by --quick);
#   * value ranges: tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) crossed with every shape level, so each supported dtype sees
#     negative, positive, dtype-extreme and degenerate ranges;
#   * layouts: non-contiguous sliced and transposed inputs must keep their exact
#     shape/stride/storage_offset while aliasing the input storage;
#   * edge cases: writing through the returned alias (mutation must be visible
#     in the original), nan/inf/±0.0 round-trip;
#   * autograd: the result is always detached — even a requires_grad input
#     yields a leaf that shares storage (broadcast/backward do not apply to a
#     unary detach-and-alias op, so they are not covered);
#   * negative: a non-tensor input raises on both the aten reference and the
#     candidate.
_DATA_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        tu.REQUIRED_DTYPES
        + utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + utils.BOOL_TYPES
        # complex32 (ComplexHalf) is intentionally left out: flag_gems'
        # comparison table has no tolerance entry for it, so it cannot be
        # validated with the shared helper even though aten::data aliases it.
        + [torch.complex64]
    )
)

# Probe on the real device: aten::data is dtype-agnostic (pure alias), but the
# probe keeps the file portable to backends where a storage dtype is missing.
_DATA_DTYPES = tu.supported_dtypes("data", candidates=_DATA_DTYPE_CANDIDATES) or [
    torch.float32
]

# fp8 cannot go through torch.testing.assert_close directly (its isclose path
# calls mul, which has no fp8 CUDA kernel); widening to float32 first is exact
# and lets the shared comparison helper run.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Representative non-contiguous layouts: strided slice (::2), offset slice
# (1:) and transpose. Each one aliases the input storage but has a layout the
# element-wise path can no longer assume contiguous.
_LAYOUT_FNS = [
    ("stride2", lambda t: t[..., ::2]),
    ("offset1", lambda t: t[..., 1:]),
    ("transpose", lambda t: t.transpose(-1, -2)),
]
_LAYOUT_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_AUTOGRAD_SHAPES = [(16, 64), (7, 13, 29)]


def _make_input(dtype, shape, value_range):
    """tu.make_input with an unsigned-dtype fallback.

    For unsigned dtypes a negative range bound clamps to 0, which can collapse
    the interval to a degenerate [0, 0]; ``make_tensor`` rejects that, so
    materialize the clamped constant locally and still exercise every spec
    range.
    """
    try:
        return tu.make_input(dtype, shape, value_range)
    except RuntimeError:
        info = torch.iinfo(dtype)
        low = max(int(tu.resolve_bound(value_range[0], dtype)), info.min)
        high = min(int(tu.resolve_bound(value_range[1], dtype)), info.max)
        if low >= high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.data callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("data", getattr(flag_gems, "data", None))


def _assert_close(res_out, ref_out, dtype):
    if dtype in _FP8_DTYPES:
        utils.gems_assert_close(res_out.float(), ref_out.float(), torch.float32)
    elif dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_alias_semantics(res_out, ref_out, inp, ref_inp, dtype):
    # The observable result must match aten exactly: same shape/dtype on the
    # same device as the input, aliasing the input storage with the identical
    # layout, and detached from autograd.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == inp.dtype
    assert res_out.device == inp.device
    assert res_out.data_ptr() == inp.data_ptr()
    assert ref_out.data_ptr() == ref_inp.data_ptr()
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()
    assert not res_out.requires_grad
    assert res_out.is_leaf
    assert res_out.grad_fn is None
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_alias_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, dtype-extreme and degenerate ranges) must
    # round-trip exactly through the aliased shallow copy for every shape level.
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_alias_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("layout", _LAYOUT_FNS, ids=[name for name, _ in _LAYOUT_FNS])
@pytest.mark.parametrize("shape", _LAYOUT_SHAPES)
@pytest.mark.parametrize("dtype", _DATA_DTYPES)
def test_data_non_contiguous(layout, shape, dtype):
    # The zero-copy alias must preserve the exact layout of a non-contiguous
    # input: shape, stride, storage offset and the shared data pointer. Slice -
    # on both the test device and the reference device so the two inputs share
    # the same memory layout.
    _, extract = layout
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = extract(base)
    ref_inp = extract(ref_base)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_alias_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_special_values(dtype):
    # data is a pure alias: +inf/-inf/nan/±0.0 round-trip unchanged; the
    # equal_nan comparison tolerates the nan value.
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(values)

    assert res_out.data_ptr() == values.data_ptr()
    # nan must compare equal to nan (the op must not sanitize it).
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.data
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_mutation(shape, dtype):
    # The result shares storage with the input: mutating through the result
    # must be visible in the original tensor. The reference runs on an
    # independent clone so the two aliases are validated separately.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    res_out = _resolve_gems_op()(inp)
    ref_out = torch.ops.aten.data(ref_inp)

    res_out.add_(1.0)
    ref_out.add_(1.0)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_close(inp, ref_inp, dtype)


@pytest.mark.data
@pytest.mark.parametrize("shape", _AUTOGRAD_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_data_autograd_detach(shape, dtype):
    # aten::data detaches from autograd: even when the input requires grad, the
    # result is a leaf that requires no grad while still aliasing the input
    # storage. There is no gradient to compute (the op is not differentiable),
    # so autograd.grad does not apply.
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp)
    if not ref_inp.requires_grad:
        ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten.data(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_alias_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark.data
def test_data_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars or strings.
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.data(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(3.14)

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.data("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()("not-a-tensor")


@pytest.mark.data
def test_data_rejects_extra_arguments():
    # aten::data takes exactly one Tensor argument; a second positional argument
    # must be rejected by the reference and by the candidate.
    inp = _make_input(torch.float32, (4, 4), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten.data(ref_inp, ref_inp)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, inp)
