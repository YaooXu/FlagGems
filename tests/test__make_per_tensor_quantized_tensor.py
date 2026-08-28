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
from . import conftest as cfg

# ``_make_per_tensor_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_tensor_quantized_tensor`` and ``-m
# _make_per_tensor_quantized_tensor`` both work.
for _name in (
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int
# zero_point) -> Tensor wraps an integer tensor (the quantized int
# representation) into a per-tensor affine quantized tensor. The output dtype is
# derived from the input dtype via toQIntType (uint8 -> quint8, int8 -> qint8,
# int32 -> qint32); no quantization arithmetic is applied -- the output's
# int_repr is an exact copy of the input values and scale/zero_point are stored
# verbatim as qparams. Inputs of any other dtype are rejected by the aten
# reference ("Creation of quantized tensor requires quantized dtype like
# torch.quint8"), so only the three integer dtypes are covered.
_MAKE_PERTENSOR_INPUT_DTYPES = [torch.uint8, torch.int8, torch.int32]
_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

# scale and zero_point are opaque qparams (the reference accepts any float /
# int), so representative values exercise the metadata path; the data path is a
# pure copy.
_MAKE_PERTENSOR_SCALES = [0.01, 0.5, 1.0]
_MAKE_PERTENSOR_ZERO_POINTS = [-1, 2]
# The shared shape set misses zero-element tensors; a pure copy kernel must
# handle the empty grid case.
_MAKE_PERTENSOR_SHAPES = utils.POINTWISE_SHAPES + [(0,)]


def _make_input(shape, dtype):
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min, info.max, shape, dtype=dtype, device=flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override installed by KernelGen for this run wins. The
    # default stays None until flag_gems._make_per_tensor_quantized_tensor is
    # registered; resolution order is: (1) override, (2) the direct flag_gems
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_make_per_tensor_quantized_tensor",
        getattr(flag_gems, "_make_per_tensor_quantized_tensor", None),
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_make_per_tensor_quantized_tensor.out",
        getattr(flag_gems, "_make_per_tensor_quantized_tensor_out", None),
    )


def _assert_quant_metadata(res_out, ref_out, ref_inp, dtype):
    # _make_per_tensor_quantized_tensor wraps integer data in a fresh quantized
    # tensor: the observable contract is the derived output dtype, the stored
    # qparams, the shape, and the int representation (an exact copy of the input
    # values). The input is never mutated and the output never aliases it.
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == _QUANT_DTYPE[dtype]
    assert res_out.shape == ref_out.shape
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    utils.gems_assert_equal(res_out.int_repr(), ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", _MAKE_PERTENSOR_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor(shape, dtype, scale, zero_point):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(
        ref_inp, scale, zero_point
    )
    res_out = _resolve_gems_op()(inp, scale, zero_point)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    # The input is only read; it must be untouched.
    utils.gems_assert_equal(inp, ref_inp)


# aten::_make_per_tensor_quantized_tensor.out(Tensor self, float scale, int
# zero_point, *, Tensor(a!) out) -> Tensor(a!) resets the qparams of the
# provided out tensor (keeping its shape and dtype) and returns the same object
# (alias semantics).
@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("shape", _MAKE_PERTENSOR_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_out(shape, dtype, scale, zero_point):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device

    # The out buffers start with different qparams so the overwrite performed by
    # the op is observable. The out dtype must already be the derived quantized
    # dtype (the out overload cannot change the out tensor's dtype).
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=_QUANT_DTYPE[dtype], device=ref_device, scale=1.0, zero_point=0
    )
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, scale, zero_point, out=ref_out_buf
    )
    assert ref_out is ref_out_buf

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    res_out = _resolve_gems_op_out()(inp, scale, zero_point, out=act_out_buf)
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_non_contiguous(dtype):
    # The copy must read through arbitrary input strides and still emit a
    # contiguous output. Slice on both the test device and the reference device
    # so the two inputs share the same memory layout.
    info = torch.iinfo(dtype)
    base = torch.randint(
        info.min, info.max, (16, 8), dtype=dtype, device=flag_gems.device
    )
    ref_base = utils.to_reference(base)
    inp = base[:, ::2]
    ref_inp = ref_base[:, ::2]

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = _resolve_gems_op()(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out, ref_inp, dtype)
    utils.gems_assert_equal(inp, ref_inp)
