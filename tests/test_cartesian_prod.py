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

# aten::cartesian_prod(Tensor[] tensors) -> Tensor takes 1-D tensors and returns
# one row per combination of one element from each input. With a single input
# this PyTorch version returns the 1-D tensor itself (shape (N,)); with k
# inputs the output is (prod(sizes), k) in row-major order (the first input
# varies slowest, like itertools.product). The op performs no arithmetic:
# values, dtype and layout are copied verbatim, so every supported dtype must
# round-trip bit-exactly and the inputs must not be mutated. Its shape-level
# dimension is therefore the *number and lengths of the 1-D inputs* rather than
# a single dense shape; the list below covers single / singleton / empty /
# equal-length / mixed-length / 3-4 input cases (the multi-input analogue of
# broadcast is combining inputs of unequal lengths, e.g. [1, 7] and [3, 1, 3]).
#
# Coverage follows the regular-operator spec adapted to this gather op:
#   * value ranges: tu.selected_ranges() over representative input lists so
#     every supported dtype round-trips negative, positive, extreme and
#     degenerate ranges exactly;
#   * shape levels: input lists selected by utils.QUICK_MODE (quick/core), from
#     a single element up to 4 inputs and empty tensors;
#   * edge cases: non-contiguous 1-D inputs, nan/inf/-inf/+-0.0 passthrough and
#     the no-mutation contract;
#   * backward: autograd.grad() against the reshape/sum analytic gradient (the
#     op is differentiable through the stack-based implementation for multiple
#     inputs; a single input is returned as-is);
#   * negative: empty list, multi-dim input, mixed input dtypes and a
#     non-tensor element all raise on the aten reference and the candidate.
# Each case below is the list of 1-D input sizes.
_CARTESIAN_PROD_SIZES = (
    [[8], [3, 5], [2, 4, 3]]
    if utils.QUICK_MODE
    else [
        [8],  # single input -> (8,)
        [1],  # single singleton input -> (1,)
        [0],  # single empty input -> (0,)
        [3, 5],  # two inputs -> (15, 2)
        [16, 16],  # two equal-length inputs -> (256, 2)
        [1, 7],  # singleton + non-singleton -> (7, 2)
        [64, 128],  # two larger inputs -> (8192, 2)
        [256, 256],  # larger two-input case -> (65536, 2)
        [2, 4, 3],  # three inputs -> (24, 3)
        [3, 1, 3],  # mixed singleton dims -> (9, 3)
        [8, 16, 32],  # larger three-input case -> (4096, 3)
        [2, 5, 8, 3],  # four inputs -> (240, 4)
        [0, 3],  # empty first input -> (0, 2)
        [5, 0],  # empty second input -> (0, 2)
    ]
)

# Representative input lists for the full value-range sweep (single / two /
# three inputs; kept small so the range sweep stays cheap).
_CARTESIAN_PROD_RANGE_SIZES = [[8], [3, 5], [2, 4, 3]]

# Backward input lists stay small (the autograd graph is built on the CPU
# reference and the analytic comparison below is per-input).
_CARTESIAN_PROD_BACKWARD_SIZES = [[8], [3, 5], [2, 4, 3], [16, 16]]

_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_INT_DTYPES = utils.ALL_INT_DTYPES
_CARTESIAN_PROD_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + utils.BOOL_TYPES


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.cartesian_prod callable, (3) None
    # when neither exists yet (the tests then fall back to the PyTorch
    # reference, keeping them runnable before an implementation is merged).
    try:
        return flag_gems.testing.resolve_gems_op(
            "cartesian_prod", getattr(flag_gems, "cartesian_prod", None)
        )
    except LookupError:
        return None


def _apply_cartesian_prod(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.cartesian_prod(inp)
    return gems_op(inp)


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _expected_grads(grad_output, sizes):
    # For input i (length n_i), the analytic gradient is grad_output[:, i] of
    # shape (prod(sizes),) viewed as (n_0, ..., n_{k-1}) and summed over every
    # dim except i: input element i appears in exactly one position along its
    # own dim of the flattened combination index. A single input returns the
    # tensor itself, so its gradient is the grad_output verbatim.
    if len(sizes) == 1:
        return [grad_output]
    grads = []
    for i in range(len(sizes)):
        view = grad_output[:, i].view(*sizes)
        dims = tuple(d for d in range(len(sizes)) if d != i)
        grads.append(view.sum(dim=dims))
    return grads


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", _CARTESIAN_PROD_SIZES)
@pytest.mark.parametrize("dtype", _CARTESIAN_PROD_DTYPES)
def test_cartesian_prod(sizes, dtype):
    # Shape levels x every supported dtype, with values from the default [-1, 1]
    # range (negative and positive for each dtype).
    inp = [tu.make_input(dtype, (size,), ["-1", "1"]) for size in sizes]
    inp_before = [t.clone() for t in inp]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = _apply_cartesian_prod(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    _assert_close(res_out, ref_out, dtype)
    # cartesian_prod is a pure gather: the inputs must not be mutated.
    for t, before in zip(inp, inp_before):
        utils.gems_assert_equal(t, utils.to_reference(before))


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", _CARTESIAN_PROD_RANGE_SIZES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES + _INT_DTYPES)
def test_cartesian_prod_value_ranges(sizes, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and degenerate constant ranges) must round-trip
    # exactly through the gather materialization.
    inp = [tu.make_input(dtype, (size,), value_range) for size in sizes]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = _apply_cartesian_prod(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", _CARTESIAN_PROD_DTYPES)
def test_cartesian_prod_non_contiguous(dtype):
    # A strided 1-D input must be read by value (indexed gather), not assumed
    # contiguous; aten and the candidate must produce identical rows.
    base = tu.make_input(dtype, (16,), ["-1", "1"])
    ref_base = utils.to_reference(base)
    other = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = [base[::2], other]
    ref_inp = [ref_base[::2], utils.to_reference(other)]
    assert not inp[0].is_contiguous()

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = _apply_cartesian_prod(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_cartesian_prod_nan_inf(dtype):
    # cartesian_prod is a pure gather: +inf/-inf/nan/+-0.0 pass through
    # unchanged (equal_nan=True is active on the float path of
    # assert_result_close).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor([1.0, -1.0], dtype=dtype, device=flag_gems.device)
    ref_inp = [utils.to_reference(values), utils.to_reference(other)]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = _apply_cartesian_prod([values, other])

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", _CARTESIAN_PROD_BACKWARD_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod_backward(sizes, dtype):
    out_shape = (sizes[0],) if len(sizes) == 1 else (math.prod(sizes), len(sizes))
    inp = [
        tu.make_input(dtype, (size,), ["-1", "1"]).requires_grad_() for size in sizes
    ]
    grad = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = [utils.to_reference(t) for t in inp]
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)
    expected = _expected_grads(ref_grad, sizes)
    for got, exp in zip(ref_in_grads, expected):
        tu.assert_result_close(got, exp)

    # The candidate forward output must match the reference...
    res_out = _apply_cartesian_prod(inp)
    _assert_close(res_out, ref_out, dtype)

    # ...and, if the candidate output is differentiable (a view of a
    # requires_grad input or an autograd-aware kernel), its gradient must match
    # the analytic value too.
    if res_out.requires_grad:
        res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
        for got, exp in zip(res_in_grads, expected):
            tu.assert_result_close(got, exp)


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, IndexError)):
            gems_op([])


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_cartesian_prod_rejects_multidim_input(dtype):
    # The op only accepts 1-D tensors; a 2-D input must raise.
    inp = tu.make_input(dtype, (3, 4), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([ref_inp])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises(RuntimeError):
            gems_op([inp])


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_mixed_dtype():
    # All inputs must share one dtype; mixing dtypes must raise.
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    b = tu.make_input(torch.int32, (4,), ["-1", "1"])
    ref_inp = [utils.to_reference(a), utils.to_reference(b)]
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod(ref_inp)
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op([a, b])


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_non_tensor():
    # The tensors argument must be a list of Tensors; a scalar element hits a
    # schema mismatch and raises.
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_inp = utils.to_reference(a)
    with pytest.raises(RuntimeError):
        torch.ops.aten.cartesian_prod([ref_inp, 3.14])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op([a, 3.14])
