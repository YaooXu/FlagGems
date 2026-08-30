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
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

from . import conftest as cfg  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_empty_affine_quantized`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# markers directly on the MarkGenerator so ``@pytest.mark._empty_affine_quantized``
# and ``-m _empty_affine_quantized`` both work.
for _name in ("_empty_affine_quantized", "_empty_affine_quantized_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )


def _resolve(name):
    # Resolved inside each test (never at import time) so that a process-local
    # override installed by KernelGen via ``override_gems_op`` for this run
    # wins. The default stays None until flag_gems registers the operator;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(name, getattr(flag_gems, name, None))


# aten::_empty_affine_quantized is a factory: given a size (plus optional
# dtype/scale/zero_point/memory_format) it returns a fresh per-tensor affine
# quantized tensor whose storage is uninitialized. The regular-operator spec
# dimensions adapt as follows:
# - Value ranges: there is no input tensor whose values could vary, so the value
#   dimension is the parameter space itself -- the quantized dtype family and
#   the qparams (scale is a double, zero_point an int64, both stored verbatim
#   without validation). QUANT_SCALES / QUANT_ZERO_POINTS below are the
#   parameter-value ranges; shape levels come from tu.selected_shapes().
# - Broadcast: N/A -- the only input is a size list; nothing to broadcast.
# - Backward: N/A -- a factory with no autograd support (no differentiable
#   input, and the uninitialized storage is never a function of another tensor).
# - nan/inf: the factory accepts and stores non-finite scale values verbatim
#   (covered by test__empty_affine_quantized_non_finite_scale below); the
#   storage bytes are uninitialized so no values can leak into them.
# - Negative cases: negative dims, non-quantized dtypes, unsupported
#   memory_formats, sparse layout and a non-quantized .out buffer must raise on
#   both the aten reference and the candidate (covered below).
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Representative qparam values spanning both signs and the zero point; these are
# the "value ranges" of the factory's parameters.
QUANT_SCALES = [-1.0, 0.0, 0.25, 1.0]
QUANT_ZERO_POINTS = [-2, 0, 3]

# Non-finite scales are stored verbatim (the factory performs no validation);
# this is the nan/inf dimension of the regular-operator spec.
NON_FINITE_SCALES = [float("nan"), float("inf"), float("-inf")]

# q_zero_point is an int64; values far outside the storage dtype range are kept
# verbatim (the aten reference stores them with no clamping), so cover one
# beyond-32-bit value to exercise the wide-int path.
WIDE_ZERO_POINTS = [1 << 40]

# torch.channels_last requires exactly 4 dims; torch.channels_last_3d exactly 5.
CHANNELS_LAST_SHAPES = [(1, 3, 8, 8), (2, 3, 16, 16), (16, 3, 32, 32)]
CHANNELS_LAST_3D_SHAPES = [(2, 3, 8, 8, 8), (4, 7, 5, 5, 5)]

# Zero-element tensors must hit the empty-grid path of the kernel.
EMPTY_SHAPES = [(0,), (0, 3)]


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _assert_quant_metadata(res_out, ref_out):
    # _empty_affine_quantized returns a fresh quantized tensor with
    # uninitialized storage: the observable contract is purely structural --
    # dtype, shape, strides, qscheme, qparams and device type. The storage bytes
    # are deliberately not compared (they may be garbage on either side).
    assert res_out.is_quantized
    assert res_out.qscheme() == torch.per_tensor_affine
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.numel() == ref_out.numel()
    assert res_out.stride() == ref_out.stride()
    # The factory stores the scale (double) and zero_point (int64) verbatim.
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized(shape, dtype, scale, zero_point):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=scale, zero_point=zero_point
    )

    gems_op = _resolve("_empty_affine_quantized")
    res_out = gems_op(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=zero_point
    )

    _assert_quant_metadata(res_out, ref_out)
    # The factory must return a fresh tensor, not a view of an internal buffer
    # (only the .out overload may return an aliased result).
    assert not res_out._is_view()
    # Default memory_format is the contiguous layout.
    assert res_out.is_contiguous()


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", NON_FINITE_SCALES)
def test__empty_affine_quantized_non_finite_scale(shape, dtype, scale):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=scale, zero_point=0
    )
    res_out = _resolve("_empty_affine_quantized")(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=0
    )

    # q_scale() round-trips the stored double; nan must compare via isnan.
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.q_zero_point() == ref_out.q_zero_point() == 0
    if math.isnan(scale):
        assert math.isnan(res_out.q_scale()) and math.isnan(ref_out.q_scale())
    else:
        assert res_out.q_scale() == ref_out.q_scale() == scale


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("zero_point", WIDE_ZERO_POINTS)
def test__empty_affine_quantized_wide_zero_point(shape, dtype, zero_point):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=1.0, zero_point=zero_point
    )
    res_out = _resolve("_empty_affine_quantized")(
        shape, dtype=dtype, device=flag_gems.device, scale=1.0, zero_point=zero_point
    )

    assert res_out.q_zero_point() == ref_out.q_zero_point() == zero_point
    assert res_out.q_scale() == ref_out.q_scale() == 1.0
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.shape == ref_out.shape == torch.Size(shape)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_empty(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device()
    )
    res_out = _resolve("_empty_affine_quantized")(
        shape, dtype=dtype, device=flag_gems.device
    )

    assert res_out.numel() == ref_out.numel() == 0
    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", CHANNELS_LAST_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_channels_last(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), memory_format=torch.channels_last
    )
    res_out = _resolve("_empty_affine_quantized")(
        shape, dtype=dtype, device=flag_gems.device, memory_format=torch.channels_last
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous(memory_format=torch.channels_last)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", CHANNELS_LAST_3D_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_channels_last_3d(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=dtype,
        device=_ref_device(),
        memory_format=torch.channels_last_3d,
    )
    res_out = _resolve("_empty_affine_quantized")(
        shape,
        dtype=dtype,
        device=flag_gems.device,
        memory_format=torch.channels_last_3d,
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous(memory_format=torch.channels_last_3d)


# aten::_empty_affine_quantized.out(SymInt[] size, *, float scale, int
# zero_point, MemoryFormat memory_format, Tensor(a!) out) -> Tensor(a!) resets
# the qparams of the provided out buffer (keeping its shape, dtype and storage)
# and returns the same object (alias semantics).
@pytest.mark._empty_affine_quantized_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized_out(shape, dtype, scale, zero_point):
    # Start the out buffer with qparams guaranteed different from every test
    # combo so the qparam reset performed by the .out overload is observable.
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=2.5, zero_point=-5
    )
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        shape, scale=scale, zero_point=zero_point, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=2.5, zero_point=-5
    )
    res_out = _resolve("_empty_affine_quantized_out")(
        shape, scale=scale, zero_point=zero_point, out=act_out_buf
    )
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.q_scale() == ref_out.q_scale() == scale
    assert res_out.q_zero_point() == ref_out.q_zero_point() == zero_point


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out_non_contiguous_view():
    # The .out overload must write qparams through a non-contiguous view of a
    # larger buffer, leaving the base tensor's own qparams untouched.
    dtype = torch.quint8
    ref_base = torch.ops.aten._empty_affine_quantized(
        (16, 8), dtype=dtype, device=_ref_device(), scale=1.0, zero_point=0
    )
    ref_sliced = ref_base[:, ::2]
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        (16, 4), scale=0.5, zero_point=3, out=ref_sliced
    )
    assert ref_out is ref_sliced

    act_base = torch.ops.aten._empty_affine_quantized(
        (16, 8), dtype=dtype, device=flag_gems.device, scale=1.0, zero_point=0
    )
    act_sliced = act_base[:, ::2]
    res_out = _resolve("_empty_affine_quantized_out")(
        (16, 4), scale=0.5, zero_point=3, out=act_sliced
    )
    assert res_out is act_sliced

    _assert_quant_metadata(res_out, ref_out)
    # The view's qparams are reset while the base keeps its original qparams.
    assert act_sliced.q_scale() == ref_sliced.q_scale() == 0.5
    assert act_sliced.q_zero_point() == ref_sliced.q_zero_point() == 3
    assert act_base.q_scale() == ref_base.q_scale() == 1.0
    assert act_base.q_zero_point() == ref_base.q_zero_point() == 0


# ---------------------------------------------------------------------------
# Negative cases: each invalid request must raise on the aten reference and the
# candidate must reject it too rather than silently succeeding.
# ---------------------------------------------------------------------------


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized_rejects_negative_size():
    with pytest.raises(RuntimeError):
        torch.ops.aten._empty_affine_quantized((-1,), dtype=torch.quint8)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve("_empty_affine_quantized")((-1,), dtype=torch.quint8)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8, torch.bool])
def test__empty_affine_quantized_rejects_non_quantized_dtype(dtype):
    # Only quantized dtypes are accepted; aten rejects others at dispatch.
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._empty_affine_quantized((2, 3), dtype=dtype)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_empty_affine_quantized")((2, 3), dtype=dtype)


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized_rejects_sparse_layout():
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._empty_affine_quantized(
            (2, 3), dtype=torch.quint8, layout=torch.sparse_coo
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_empty_affine_quantized")(
            (2, 3), dtype=torch.quint8, layout=torch.sparse_coo
        )


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.channels_last_3d],
)
def test__empty_affine_quantized_rejects_invalid_memory_format(memory_format):
    # preserve_format has no meaning for a factory (there is no input whose
    # format could be preserved) and channels_last_3d needs 5 dims; aten rejects
    # both for a rank-4 request.
    if memory_format == torch.channels_last_3d:
        with pytest.raises((RuntimeError, TypeError)):
            torch.ops.aten._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _resolve("_empty_affine_quantized")(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
    else:
        with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
            torch.ops.aten._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
        with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
            _resolve("_empty_affine_quantized")(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out_rejects_non_quantized_buffer():
    ref_buf = torch.empty((2, 3), dtype=torch.float32, device=_ref_device())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._empty_affine_quantized.out((2, 3), out=ref_buf)

    act_buf = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_empty_affine_quantized_out")((2, 3), out=act_buf)


@pytest.mark._empty_affine_quantized_out
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="CPU reference resizes the out buffer; only the CUDA reference rejects "
    "a .out size that does not match the buffer (resize_ is unimplemented on "
    "QuantizedCUDA)",
)
def test__empty_affine_quantized_out_rejects_shape_mismatch():
    buf = torch.ops.aten._empty_affine_quantized(
        (2, 3), dtype=torch.quint8, device=flag_gems.device
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._empty_affine_quantized.out((4, 6), out=buf)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_empty_affine_quantized_out")((4, 6), out=buf)
