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

_ABS_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_ABS_INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
_ABS_SIGNED_INT_DTYPES = [d for d in _ABS_INT_DTYPES if d.is_signed]
_ABS_DTYPES = _ABS_FLOAT_DTYPES + _ABS_INT_DTYPES + utils.BOOL_TYPES

# Shapes that exercise 0-dim scalars, degenerate/empty tensors and
# non-contiguous strides (the pointwise kernel must honor the input strides).
_ABS_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_ABS_NONCONTIG_SHAPES = [(17, 33), (5, 7, 9)]

# Backward shapes stay small (the autograd graph is built on the CPU reference
# and the analytic comparison below is elementwise).
_ABS_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _make_input(dtype, shape, value_range):
    return tu.make_input(dtype, shape, value_range)


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op("abs", flag_gems.abs)


def _resolve_gems_op_inplace():
    return flag_gems.testing.resolve_gems_op("abs_", flag_gems.abs_)


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op("abs", getattr(flag_gems, "abs", None))


@pytest.mark.abs
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_FLOAT_DTYPES)
def test_abs_float_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_INT_DTYPES + utils.BOOL_TYPES)
def test_abs_int_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    # int/bool abs is exact: assert_result_close uses an atol=0/rtol=0 path.
    tu.assert_result_close(res_out, ref_out)


if tu.LEVEL == "all":

    @pytest.mark.abs
    @pytest.mark.parametrize("dtype", _ABS_FLOAT_DTYPES)
    def test_abs_nan_inf(dtype):
        # inf/-inf -> +inf, nan -> nan, -0.0 -> 0.0. 1e30/-1e30 also cover the
        # overflow-to-inf path in fp16 (1e30 remains finite in bf16); equal_nan=True tolerates nan outputs.
        inp = torch.tensor(
            [
                float("inf"),
                float("-inf"),
                float("nan"),
                0.0,
                -0.0,
                1.5,
                -2.5,
                1e30,
                -1e30,
            ],
            dtype=dtype,
            device=flag_gems.device,
        )
        ref_inp = tu.to_reference(inp)

        ref_out = torch.ops.aten.abs(ref_inp)
        res_out = _resolve_gems_op()(inp)

        tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", _ABS_SIGNED_INT_DTYPES)
def test_abs_int_min_stays(dtype):
    # |INT_MIN| == INT_MIN in PyTorch (no wrap-around); pin this contract.
    # Unsigned dtypes have no negative minimum, so only the signed path is
    # meaningful here.
    min_val = torch.iinfo(dtype).min
    inp = torch.tensor(
        [min_val, min_val + 1, 0, 1, -1], dtype=dtype, device=flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", _ABS_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", _ABS_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_noncontiguous(shape, dtype):
    # transposed views have non-unit strides; the kernel must honor them.
    inp = _make_input(dtype, shape, ["-1", "1"]).transpose(-1, -2)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.abs(ref_inp)
    res_out = _resolve_gems_op()(inp)

    tu.assert_result_close(res_out, ref_out)


if tu.LEVEL == "all":

    @pytest.mark.abs
    @pytest.mark.parametrize("shape", _ABS_BACKWARD_SHAPES)
    @pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
    def test_abs_backward(shape, dtype):
        inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
        grad = _make_input(dtype, shape, ["-1", "1"])
        ref_inp = tu.to_reference(inp)
        ref_grad = tu.to_reference(grad)

        ref_out = torch.ops.aten.abs(ref_inp)
        ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

        # d|x|/dx == sign(x) (torch defines sign(0) == 0), so the reference
        # gradient must match the analytic value; this validates the reference
        # autograd path itself.
        expected_in_grad = torch.sign(ref_inp) * ref_grad
        tu.assert_result_close(ref_in_grad, expected_in_grad)

        # The candidate forward output must match the reference...
        res_out = _resolve_gems_op()(inp)
        tu.assert_result_close(res_out, ref_out)

        assert res_out.requires_grad
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark.abs_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs__value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp.clone())

    ref_out = torch.ops.aten.abs_(ref_inp)
    res_out = _resolve_gems_op_inplace()(inp)

    # In-place semantics: the call returns the mutated input tensor itself.
    assert res_out is inp
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.abs_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ABS_DTYPES)
def test_abs_out(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.abs.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is res_out
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.abs_negative
def test_abs_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.abs(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.abs_negative
def test_abs_rejects_string():
    # A non-numeric, non-tensor argument must be rejected, not coerced.
    with pytest.raises((TypeError, RuntimeError)):
        torch.ops.aten.abs("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()("not-a-tensor")
