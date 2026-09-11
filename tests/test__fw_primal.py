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

# ``_fw_primal`` starts with an underscore and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names, so register it on the
# MarkGenerator directly (``@pytest.mark._fw_primal`` and ``-m _fw_primal``).
setattr(
    pytest.mark,
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a) is the forward-mode
# AD view primitive: it returns an aliasing view of ``self`` that shares the
# input storage (same shape, strides, storage offset, data_ptr and dtype)
# without any arithmetic. Coverage adapts the regular-operator spec to a pure
# metadata view:
#   * shapes:       tu.selected_shapes() (ranks 0-5 at the full level);
#   * value ranges: tu.selected_ranges() (the five spec ranges), so every
#                   supported dtype round-trips negative, positive, extreme and
#                   degenerate value windows bit-for-bit;
#   * dtypes:       the spec's required dtypes plus float64/complex/bool, probed
#                   on the active device (a pure view accepts every storage
#                   dtype, fp8 included);
#   * levels:       the documented level 0 plus higher levels, which aten also
#                   accepts for plain tensors with no registered tangent;
#   * edge cases:   non-contiguous strided inputs, empty tensors, nan/inf/+-0.0,
#                   aliasing mutation through the returned view, and an
#                   autograd/backward gradient check;
#   * negative:     non-tensor input, non-int level and a missing level must be
#                   rejected.
# There is no broadcast dimension: the operator is unary and performs no
# arithmetic, so there is nothing to broadcast against.

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _probe_dtype(dtype):
    """Return True if the reference op accepts this storage dtype on device."""
    try:
        probe_inp = torch.testing.make_tensor(
            (4,), dtype=dtype, device=flag_gems.device, low=0, high=1
        )
        torch.ops.aten._fw_primal(probe_inp, 0)
        return True
    except Exception:
        return False


_FW_PRIMAL_CANDIDATE_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)
# tu.supported_dtypes() cannot probe this op with its default path (the
# ``default`` overload needs the extra ``level`` argument), so hand it a
# two-argument probe and keep the static list as a fallback.
_FW_PRIMAL_DTYPES = (
    tu.supported_dtypes(
        "_fw_primal",
        _FW_PRIMAL_CANDIDATE_DTYPES,
        probe=lambda _op, dtype: _probe_dtype(dtype),
    )
    or _FW_PRIMAL_CANDIDATE_DTYPES
)

# ``level`` is the forward-AD level: 0 is the documented level, while 1/3
# validate that a plain tensor without a registered tangent still round-trips.
_FW_PRIMAL_LEVELS = [0, 1, 3]
_FW_PRIMAL_LEVEL_SHAPES = [(), (256,), (7, 13, 29)]

_FW_PRIMAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_FW_PRIMAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_FW_PRIMAL_EMPTY_SHAPES = [(0,), (2, 0, 3)]
_FW_PRIMAL_BACKWARD_SHAPES = [(), (256,), (7, 13, 29)]
_FW_PRIMAL_BACKWARD_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
_FW_PRIMAL_SPECIAL_VALUES = [
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    1.5,
    -1.5,
    float("nan"),
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so the process-local
    # override installed by KernelGen for this run wins. Resolution order is:
    # (1) override_gems_op, (2) flag_gems._fw_primal, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_fw_primal", getattr(flag_gems, "_fw_primal", None)
    )


def _make_input(dtype, shape, value_range):
    """tu.make_input with a fallback for a window the dtype cannot represent.

    The spec's [-1, 0] range clamps to low == high == 0 for unsigned dtypes and
    ``make_tensor`` rejects that degenerate window, although an all-zero tensor
    is exactly what the range means for uint8.
    """
    try:
        return tu.make_input(dtype, shape, value_range)
    except RuntimeError:
        return torch.zeros(shape, dtype=dtype, device=flag_gems.device)


def _assert_values(res_out, ref_out):
    if ref_out.dtype in _FP8_DTYPES:
        # torch.testing.assert_close has no tolerance implementation for fp8
        # storage; a pure view round-trips the payload bit-for-bit.
        utils.gems_assert_equal(res_out, ref_out)
    else:
        tu.assert_result_close(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # _fw_primal returns an aliasing view (Tensor(a)): the observable layout
    # must match aten exactly and the result must share the input storage.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal(shape, value_range, dtype):
    # The full shape x value-range x dtype grid at the documented level 0. A
    # view never inspects or transforms the stored values, so every range must
    # round-trip exactly.
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    _assert_values(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_LEVEL_SHAPES)
@pytest.mark.parametrize("level", _FW_PRIMAL_LEVELS)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_level(shape, level, dtype):
    # The ``level`` argument is orthogonal to the shape/value grid, so sweep it
    # over representative ranks (0-dim, 1-dim, 3-dim) for every dtype.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    _assert_values(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_non_contiguous(shape, level, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    _assert_values(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_MUTATION_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test__fw_primal_mutation(shape, dtype):
    # The result is a true alias of the input: writing through the returned view
    # must be observable on the candidate-side input tensor, and the reference
    # must behave identically. The reference runs on an independent clone so the
    # two aliases are validated separately.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    if dtype == torch.bool:
        res_out.fill_(True)
        ref_out.fill_(True)
    elif dtype.is_floating_point:
        res_out.fill_(2.5)
        ref_out.fill_(2.5)
    else:
        res_out.fill_(7)
        ref_out.fill_(7)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__fw_primal_special_values(dtype):
    # A pure view preserves every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    values = torch.tensor(
        _FW_PRIMAL_SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(values.clone())

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(values, 0)

    _assert_view_semantics(res_out, ref_out, values)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_empty(shape, dtype):
    # Empty tensors (0 elements) still carry a valid layout; the view must
    # preserve shape, strides, storage offset and data_ptr exactly.
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    _assert_values(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_BACKWARD_DTYPES)
def test__fw_primal_backward(shape, dtype):
    # A view is transparent to autograd: the gradient of a loss built on the
    # result must match the reference gradient (the view contributes identity).
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    (ref_grad,) = torch.autograd.grad((ref_out.float() ** 2).sum(), ref_inp)
    (res_grad,) = torch.autograd.grad((res_out.float() ** 2).sum(), inp)

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python float hits the invalid
    # argument path and raises. The candidate must fail too rather than silently
    # accept scalars. LookupError is tolerated so the file still runs without an
    # injected override.
    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(3.14, 0)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, LookupError)
    ):
        _resolve_gems_op()(3.14, 0)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_int_level():
    # ``level`` is an int in the schema; a float is a cast error at the
    # dispatcher boundary and must be rejected by the candidate as well.
    inp = _make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(ref_inp, 1.5)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, LookupError)
    ):
        _resolve_gems_op()(inp, 1.5)


@pytest.mark._fw_primal
def test__fw_primal_rejects_missing_level():
    # ``level`` has no default in the schema; omitting it must fail on both the
    # reference and the candidate instead of silently using level 0.
    inp = _make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(ref_inp)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, LookupError)
    ):
        _resolve_gems_op()(inp)
