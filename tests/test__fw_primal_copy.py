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

# ``_fw_primal_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._fw_primal_copy`` and ``-m
# _fw_primal_copy`` both work.
setattr(
    pytest.mark,
    "_fw_primal_copy",
    MarkDecorator(Mark("_fw_primal_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_fw_primal_copy_out",
    MarkDecorator(Mark("_fw_primal_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_fw_primal_copy(Tensor self, int level) -> Tensor materializes the
# primal of a dual tensor at forward-mode AD level ``level`` as a fresh
# contiguous copy. For a plain tensor the primal is the tensor itself, so the
# op is a pure elementwise copy that must not alias or mutate the input. The
# native CompositeExplicitAutograd implementation is only reachable through the
# dispatcher for inference tensors (it is guarded by an InferenceMode assert),
# so the reference must run inside torch.inference_mode(). In that context aten
# ignores ``level`` and returns the copy for any level, which the candidate
# must reproduce.
_FW_PRIMAL_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
_FW_PRIMAL_COPY_LEVELS = [0, 1]


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_fw_primal_copy" and "_fw_primal_copy.out".
    return flag_gems.testing.resolve_gems_op(
        "_fw_primal_copy", getattr(flag_gems, "_fw_primal_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_fw_primal_copy.out", getattr(flag_gems, "_fw_primal_copy_out", None)
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype):
    # _fw_primal_copy returns a fresh contiguous copy: same shape/dtype, no
    # aliasing of the input, and the input is never mutated.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp, ref_inp)
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._fw_primal_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_COPY_DTYPES)
@pytest.mark.parametrize("level", _FW_PRIMAL_COPY_LEVELS)
def test__fw_primal_copy(shape, dtype, level):
    with torch.inference_mode():
        inp = _make_input(shape, dtype)
        ref_inp = utils.to_reference(inp.clone())

        ref_out = torch.ops.aten._fw_primal_copy(ref_inp, level)
        res_out = _resolve_gems_op()(inp, level)

        _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)


@pytest.mark._fw_primal_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_COPY_DTYPES)
def test__fw_primal_copy_out(shape, dtype):
    with torch.inference_mode():
        inp = _make_input(shape, dtype)
        ref_inp = utils.to_reference(inp.clone())
        out = torch.empty_like(inp)
        ref_out = torch.empty_like(ref_inp)

        ref_ret = torch.ops.aten._fw_primal_copy.out(ref_inp, 0, out=ref_out)
        res_ret = _resolve_gems_op_out()(inp, 0, out=out)

        # The .out variant must write into and return the out tensor itself.
        assert res_ret is out
        assert ref_ret is ref_out
        _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp, dtype)
        utils.gems_assert_equal(out, ref_out)


@pytest.mark._fw_primal_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__fw_primal_copy_special_values(dtype):
    # A pure copy preserves every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    with torch.inference_mode():
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
            dtype=dtype,
            device=flag_gems.device,
        )
        ref_inp = utils.to_reference(values.clone())

        ref_out = torch.ops.aten._fw_primal_copy(ref_inp, 0)
        res_out = _resolve_gems_op()(values, 0)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._fw_primal_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__fw_primal_copy_non_contiguous(dtype):
    # The copy must read through arbitrary input strides and still emit a
    # contiguous output. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    with torch.inference_mode():
        base = torch.randn((8, 8, 8), dtype=dtype, device=flag_gems.device)
        ref_base = utils.to_reference(base)
        inp = base[:, ::2, ::2]
        ref_inp = ref_base[:, ::2, ::2].clone()

        ref_out = torch.ops.aten._fw_primal_copy(ref_inp, 0)
        res_out = _resolve_gems_op()(inp, 0)

        _assert_copy_semantics(res_out, ref_out, inp, ref_inp, dtype)
