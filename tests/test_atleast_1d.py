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

# aten::atleast_1d is a pure view/identity op: 0-dim tensors are reshaped to
# (1,) (a view) while tensors with one or more dimensions are returned as-is.
# No arithmetic is performed, so the result must match bit-for-bit, alias the
# input, and every storage dtype the op supports is covered (float / int /
# bool; complex is skipped per the value-range spec's dtype list). The .default
# overload is resolved through its public name "atleast_1d" (KernelGen's
# override_gems_op("atleast_1d", ...) wins over the direct callable); the
# .Sequence overload shares the same public name.
_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_INT_DTYPES = utils.ALL_INT_DTYPES
_ATLEAST_1D_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + utils.BOOL_TYPES

# Shapes always include the 0-dim scalar, which is atleast_1d's defining case
# (scalar -> (1,) view); the rest follow the shared shape-level selection.
_ATLEAST_1D_SHAPES = list(tu.selected_shapes())
if () not in _ATLEAST_1D_SHAPES:
    _ATLEAST_1D_SHAPES.insert(0, ())

# Backward shapes stay small (the autograd graph is built on the CPU reference
# and the analytic comparison below is elementwise).
_ATLEAST_1D_BACKWARD_SHAPES = [(), (16, 64), (7, 13, 29)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.atleast_1d callable, (3) None when
    # neither exists yet (the tests then fall back to the PyTorch reference).
    try:
        return flag_gems.testing.resolve_gems_op(
            "atleast_1d", getattr(flag_gems, "atleast_1d", None)
        )
    except LookupError:
        return None


def _apply_atleast_1d(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.atleast_1d(inp)
    return gems_op(inp)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_atleast_1d_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # atleast_1d is a view op: the result must alias the input.
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INT_DTYPES + utils.BOOL_TYPES)
def test_atleast_1d_int_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # atleast_1d is a view op: the result must alias the input.
    assert res_out.data_ptr() == inp.data_ptr()
    # int/bool atleast_1d is exact: assert_result_close uses atol=0/rtol=0.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_atleast_1d_nan_inf(dtype):
    # atleast_1d is a view: nan/inf/-inf/+-0.0 must pass through bit-for-bit.
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.data_ptr() == inp.data_ptr()
    # equal_nan=True is active on the float path of assert_result_close.
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_1d
@pytest.mark.parametrize("shape", _ATLEAST_1D_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_1d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    # atleast_1d's output is shape (1,) for a 0-dim input and the input shape
    # otherwise, so the grad_output must be built against the output shape.
    out_shape = (1,) if shape == () else shape
    grad = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.atleast_1d(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    # atleast_1d is a reshape-view: d(out)/d(in) is the identity, so the
    # gradient is the grad_output squeezed back to the input shape. This
    # validates the reference autograd path itself.
    expected_in_grad = ref_grad.reshape(ref_inp.shape)
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward output must match the reference...
    res_out = _apply_atleast_1d(inp)
    tu.assert_result_close(res_out, ref_out)

    # ...and, if the candidate output is differentiable (the reference fallback
    # returns a view of a requires_grad input), its gradient must match too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark.atleast_1d_sequence
@pytest.mark.parametrize("shape", _ATLEAST_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ATLEAST_1D_DTYPES)
def test_atleast_1d_sequence(shape, value_range, dtype):
    # Mix a 0-dim scalar with the current shape so the sequence overload
    # exercises both the scalar -> (1,) view path and the identity path.
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, shape, value_range),
        tu.make_input(dtype, shape, value_range),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_1d.Sequence(ref_inp)
    res_out = _apply_atleast_1d(inp)

    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        # atleast_1d is a view op: each result must alias its input.
        assert res.data_ptr() == src.data_ptr()
        tu.assert_result_close(res, ref)


@pytest.mark.atleast_1d_negative
def test_atleast_1d_rejects_non_tensor():
    # The aten op only accepts a Tensor (a list of Tensors goes through the
    # .Sequence overload); Python scalars hit a schema mismatch and raise.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_1d(3.14)
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(3.14)
