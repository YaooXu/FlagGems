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

# SPDX-License-Identifier: Apache-2.0
import os
import sys

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package (and the sibling ``benchmark`` package) resolve to THIS
# checkout no matter how pytest is invoked.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_make_per_channel_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_channel_quantized_tensor`` and ``-m
# _make_per_channel_quantized_tensor`` both work.
for _name in (
    "_make_per_channel_quantized_tensor",
    "_make_per_channel_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )


def _resolve(name):
    # Resolved inside each test (never at module import time) so that a
    # process-local override installed by KernelGen via ``override_gems_op`` for
    # this run wins. The .out overload is resolved through its aten schema name
    # ("_make_per_channel_quantized_tensor.out") so the override key matches the
    # schema; its direct fallback callable carries the flag_gems underscore
    # suffix ("_make_per_channel_quantized_tensor_out"). Resolution order:
    # (1) override, (2) the direct flag_gems callable, (3) LookupError.
    if name == "_make_per_channel_quantized_tensor.out":
        default = getattr(flag_gems, "_make_per_channel_quantized_tensor_out", None)
    else:
        default = getattr(flag_gems, name, None)
    return flag_gems.testing.resolve_gems_op(name, default)


# aten::_make_per_channel_quantized_tensor(self, scale, zero_point, axis)
# reinterprets a *plain integer* tensor (uint8/int8/int32) as the underlying
# storage of a torch.per_channel_affine quantized tensor carrying the given
# per-channel scale/zero_point metadata. The output quantized dtype is derived
# from the input storage dtype: uint8 -> quint8, int8 -> qint8, int32 -> qint32;
# the integer values are copied to the output storage unchanged. The data path
# is a pure bit-exact copy and the metadata is stored verbatim, so every
# assertion is an exact-equality check (utils.gems_assert_equal). There is no
# broadcast dimension (the op takes a single storage tensor plus metadata) and
# no autograd support (quantized tensors), so those spec dimensions do not
# apply here; the value-range dimension is applied to the storage tensor and to
# the scales/zero_points metadata, which are the only value-carrying inputs.
STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]

_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

# (shape, axis) pairs covering rank 1-5, every valid axis position, both
# positive and negative axis encodings, and the core shape levels of the shared
# table ((256,) and (7, 13, 29)). num_channels = shape[axis] varies so the
# per-channel metadata length is exercised. 0-dim shapes and out-of-range axes
# are deliberately absent: the CPU reference rejects them while the CUDA
# reference accepts them, so neither can be asserted uniformly across backends.
SHAPE_AXIS = (
    [((2, 3, 4), 1)]
    if utils.QUICK_MODE
    else [
        ((7,), 0),
        ((7,), -1),
        ((256,), 0),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((2, 3, 4), 1),
        ((2, 3, 4), -2),
        ((7, 13, 29), 2),
        ((2, 3, 4, 5), 2),
        ((2, 3, 4, 5), -1),
        ((16, 32, 64), 1),
    ]
)

# The quantizer stores scales as float64 regardless of the input dtype; cover
# both supported scale dtypes.
SCALE_DTYPES = [torch.float32, torch.float64]


def _storage_ranges(storage_dtype):
    # The shared value-range table includes ["-1", "0"], whose lower bound
    # resolves negative for every dtype. torch.testing.make_tensor rejects that
    # range for the unsigned uint8 storage ("random_ expects 'from' to be less
    # than 'to', but got from=0 >= to=0"), so it is dropped for uint8 only; the
    # signed storages keep the full table.
    if storage_dtype == torch.uint8:
        return [r for r in tu.selected_ranges() if r != ["-1", "0"]]
    return tu.selected_ranges()


STORAGE_RANGE_COMBOS = [
    (storage_dtype, value_range)
    for storage_dtype in STORAGE_DTYPES
    for value_range in _storage_ranges(storage_dtype)
]


def _make_storage_input(shape, storage_dtype, value_range):
    # tu.make_input rerouted to flag_gems.device (the shared helper allocates there
    # by default).
    low = tu.resolve_bound(value_range[0], storage_dtype)
    high = tu.resolve_bound(value_range[1], storage_dtype)
    if not storage_dtype.is_floating_point:
        low, high = int(low), int(high)
    if low == high:
        return torch.full(shape, low, dtype=storage_dtype, device=flag_gems.device)
    return torch.testing.make_tensor(
        shape, dtype=storage_dtype, device=flag_gems.device, low=low, high=high
    )


def _make_metadata(shape, axis, storage_dtype, scale_dtype):
    # Positive scales, and zero_points spanning each storage dtype's valid range
    # (the CPU per-channel affine validation checks the zero_point range against
    # the quantized dtype, so they must stay inside it).
    num_channels = shape[axis]
    scales = torch.rand(num_channels, dtype=scale_dtype, device=flag_gems.device) + 0.1
    if storage_dtype == torch.uint8:
        low, high = 0, 256
    elif storage_dtype == torch.int8:
        low, high = -128, 128
    else:
        low, high = torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
    zero_points = torch.randint(
        low, high, (num_channels,), dtype=storage_dtype, device=flag_gems.device
    )
    return scales, zero_points


def _assert_per_channel_quantized(
    res_out, ref_out, inp, scales, zero_points, axis, equal_nan=False
):
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.numel() == ref_out.numel()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.qscheme() == ref_out.qscheme() == torch.per_channel_affine
    assert res_out.q_per_channel_axis() == ref_out.q_per_channel_axis() == axis

    # The getters always report float64 scales and int64 zero_points.
    assert (
        res_out.q_per_channel_scales().dtype
        == ref_out.q_per_channel_scales().dtype
        == torch.float64
    )
    assert (
        res_out.q_per_channel_zero_points().dtype
        == ref_out.q_per_channel_zero_points().dtype
        == torch.int64
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        ref_out.q_per_channel_scales(),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        ref_out.q_per_channel_zero_points(),
        equal_nan=equal_nan,
    )

    # The stored metadata must reproduce the caller-supplied values exactly
    # (scales widened to float64, zero_points to int64).
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(),
        utils.to_reference(scales).to(torch.float64),
        equal_nan=equal_nan,
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        utils.to_reference(zero_points).to(torch.int64),
        equal_nan=equal_nan,
    )

    # The integer storage is copied unchanged from the input tensor, so the
    # underlying representation must match the input and the reference
    # bit-exactly.
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(inp))
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


def _assert_per_channel_float_qparams(res_out, ref_out, inp, scales, zero_points, axis):
    # A *floating-point* zero_point tensor switches the output qscheme to
    # torch.per_channel_affine_float_qparams, whose getters report float32
    # scales and zero_points (unlike the int64-zero_point path above).
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert (
        res_out.qscheme() == ref_out.qscheme() == torch.per_channel_affine_float_qparams
    )
    assert res_out.q_per_channel_axis() == ref_out.q_per_channel_axis() == axis

    assert (
        res_out.q_per_channel_scales().dtype
        == ref_out.q_per_channel_scales().dtype
        == torch.float32
    )
    assert (
        res_out.q_per_channel_zero_points().dtype
        == ref_out.q_per_channel_zero_points().dtype
        == torch.float32
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(), ref_out.q_per_channel_scales()
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(), ref_out.q_per_channel_zero_points()
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_scales(), utils.to_reference(scales).to(torch.float32)
    )
    utils.gems_assert_equal(
        res_out.q_per_channel_zero_points(),
        utils.to_reference(zero_points).to(torch.float32),
    )
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(inp))
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype,value_range", STORAGE_RANGE_COMBOS)
def test__make_per_channel_quantized_tensor_value_ranges(
    shape, axis, storage_dtype, value_range
):
    # The storage tensor and the per-channel metadata are the only value-
    # carrying inputs. Feed all three from the shared value-range table so every
    # numeric range (negative, positive, full dtype bounds, degenerate
    # constants) is exercised; the copy and the metadata must be preserved
    # bit-exactly regardless of sign, magnitude or dtype bounds. The scales use
    # float64 (the canonical quantizer storage dtype) with the full range table,
    # including nan-free extremes like +/-finfo(float64).max, which the
    # reference stores verbatim.
    inp = _make_storage_input(shape, storage_dtype, value_range)
    num_channels = shape[axis]
    scales = tu.make_input(torch.float64, (num_channels,), value_range).to(
        flag_gems.device
    )
    zero_points = tu.make_input(storage_dtype, (num_channels,), value_range).to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test__make_per_channel_quantized_tensor(shape, axis, storage_dtype, scale_dtype):
    # The default overload over a representative value range, covering both
    # input scale dtypes: the quantizer must canonicalize float32 and float64
    # scales to float64 exactly (no lossy round-trip through a wider type).
    inp = _make_storage_input(shape, storage_dtype, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis)
    # The input is only read; it must be untouched.
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test__make_per_channel_quantized_tensor_non_contiguous(storage_dtype, scale_dtype):
    # A transposed view whose strides do not match the contiguous layout: the
    # reference CPU kernel materializes a contiguous copy and the CUDA kernel
    # iterates over the input strides, so the logical (not physical) values
    # must be preserved in the output storage.
    inp = _make_storage_input((4, 3, 8), storage_dtype, ["0", "max"]).transpose(0, 1)
    assert not inp.is_contiguous()  # shape (3, 4, 8)
    axis = 1
    scales, zero_points = _make_metadata(inp.shape, axis, storage_dtype, scale_dtype)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_quantized(res_out, ref_out, inp, scales, zero_points, axis)


# aten::_make_per_channel_quantized_tensor.out(Tensor self, Tensor scale, Tensor
# zero_point, int axis, *, Tensor(a!) out) -> Tensor(a!) overwrites the
# quantizer metadata and the storage of the provided out tensor (keeping its
# qscheme and dtype) and returns the same object (alias semantics). The out
# buffer must already carry the derived quantized dtype.
@pytest.mark._make_per_channel_quantized_tensor_out
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_out(shape, axis, storage_dtype):
    inp = _make_storage_input(shape, storage_dtype, ["0", "max"])
    scales, zero_points = _make_metadata(shape, axis, storage_dtype, torch.float32)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    quantized_dtype = _QUANT_DTYPE[storage_dtype]

    # The buffer is created with deliberately *different* metadata so the
    # overwrite performed by the op is observable. The .out variant keeps the
    # out buffer dtype, which must already be the derived quantized dtype.
    ref_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=utils.to_reference(
            torch.full(
                (shape[axis],), 9.0, dtype=torch.float64, device=flag_gems.device
            )
        ),
        zero_points=utils.to_reference(
            torch.full((shape[axis],), 9, dtype=torch.int64, device=flag_gems.device)
        ),
        axis=axis,
        dtype=quantized_dtype,
        device=ref_device,
    )
    ref_ret = torch.ops.aten._make_per_channel_quantized_tensor.out(
        ref_inp, ref_scales, ref_zero_points, axis, out=ref_out_buf
    )
    assert ref_ret is ref_out_buf

    act_out_buf = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full(
            (shape[axis],), 9.0, dtype=torch.float64, device=flag_gems.device
        ),
        zero_points=torch.full(
            (shape[axis],), 9, dtype=torch.int64, device=flag_gems.device
        ),
        axis=axis,
        dtype=quantized_dtype,
        device=flag_gems.device,
    )
    res_ret = _resolve("_make_per_channel_quantized_tensor.out")(
        inp, scales, zero_points, axis, out=act_out_buf
    )
    assert res_ret is act_out_buf

    _assert_per_channel_quantized(
        act_out_buf, ref_out_buf, inp, scales, zero_points, axis
    )
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", SHAPE_AXIS)
@pytest.mark.parametrize("storage_dtype", STORAGE_DTYPES)
def test__make_per_channel_quantized_tensor_float_zero_points(
    shape, axis, storage_dtype
):
    # A floating-point zero_point tensor selects the
    # per_channel_affine_float_qparams scheme (float32 scales/zero_points
    # stored as-is) instead of the int64-zero_point per_channel_affine scheme.
    inp = _make_storage_input(shape, storage_dtype, ["0", "max"])
    num_channels = shape[axis]
    scales = (
        torch.rand(num_channels, dtype=torch.float32, device=flag_gems.device) + 0.1
    )
    zero_points = torch.rand(num_channels, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, axis
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, axis
    )

    _assert_per_channel_float_qparams(res_out, ref_out, inp, scales, zero_points, axis)


@pytest.mark._make_per_channel_quantized_tensor
def test__make_per_channel_quantized_tensor_nan_inf_scales():
    # nan/inf/-inf scales are accepted by both references and stored verbatim.
    # equal_nan is required because the exact-equality helpers compare
    # nan != nan by default.
    shape = (2, 3, 4)
    scales = torch.tensor(
        [float("nan"), float("inf"), -float("inf")],
        dtype=torch.float64,
        device=flag_gems.device,
    )
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    inp = _make_storage_input(shape, torch.uint8, ["0", "1"])
    ref_inp = utils.to_reference(inp)
    ref_scales = utils.to_reference(scales)
    ref_zero_points = utils.to_reference(zero_points)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        ref_inp, ref_scales, ref_zero_points, 1
    )

    res_out = _resolve("_make_per_channel_quantized_tensor")(
        inp, scales, zero_points, 1
    )

    _assert_per_channel_quantized(
        res_out, ref_out, inp, scales, zero_points, 1, equal_nan=True
    )


# --- Negative cases ----------------------------------------------------------
# Each test asserts the reference rejects the arguments first (documenting that
# the case is genuinely invalid) and then that the candidate rejects them
# identically. Only cases that both the CPU and the CUDA reference reject are
# listed here: out-of-range axes, out-of-range zero_points and 0-dim shapes are
# rejected by the CPU reference but *accepted* by the CUDA reference, so they
# cannot be asserted uniformly and are deliberately not negative cases.


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("invalid_dtype", [torch.float32, torch.int64])
def test__make_per_channel_quantized_tensor_negative_invalid_input_dtype(
    invalid_dtype,
):
    # Only integer storage dtypes (uint8/int8/int32) can be reinterpreted as
    # quantized storage; float and int64 inputs are rejected.
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=invalid_dtype, device=flag_gems.device)
    scales = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_dtype", [torch.int32, torch.int64])
def test__make_per_channel_quantized_tensor_negative_non_float_scales(scale_dtype):
    # The scale tensor must be floating point; integer scales are rejected.
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.tensor([1, 2, 3], dtype=scale_dtype, device=flag_gems.device)
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
def test__make_per_channel_quantized_tensor_negative_scale_not_1d():
    # The per-channel metadata must be 1-D (one entry per channel); a 2-D scale
    # tensor is rejected.
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(2, 3, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)


@pytest.mark._make_per_channel_quantized_tensor
@pytest.mark.parametrize("scale_len,zero_point_len", [(2, 3), (3, 2), (0, 3), (3, 0)])
def test__make_per_channel_quantized_tensor_negative_metadata_length_mismatch(
    scale_len, zero_point_len
):
    # The factory requires scales.numel() == zero_points.numel(); the lengths
    # only need to match each other (they are not checked against size[axis]).
    shape = (2, 3)
    inp = torch.zeros(shape, dtype=torch.uint8, device=flag_gems.device)
    scales = torch.rand(scale_len, dtype=torch.float32, device=flag_gems.device)
    zero_points = torch.zeros(
        zero_point_len, dtype=torch.int64, device=flag_gems.device
    )

    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._make_per_channel_quantized_tensor(
            utils.to_reference(inp),
            utils.to_reference(scales),
            utils.to_reference(zero_points),
            1,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve("_make_per_channel_quantized_tensor")(inp, scales, zero_points, 1)
