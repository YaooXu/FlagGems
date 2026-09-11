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

"""Correctness tests for ``aten::cartesian_prod(Tensor[] tensors) -> Tensor``.

``cartesian_prod`` consumes a list of 1-D tensors and writes one row per
combination of one element from each input (``prod(sizes)`` rows, in row-major
order with the first input varying slowest, i.e. ``itertools.product`` order).
With a single input PyTorch returns the 1-D tensor itself (shape ``(N,)``); with
``k`` inputs the output is ``(prod(sizes), k)``.

The op performs no arithmetic: values, dtype and layout are copied verbatim.
Consequences for the regular-operator spec:

* its shape-level dimension is the *number and lengths of the 1-D inputs*
  rather than a single dense shape, so the spec's seven dense shapes are
  represented here by the input-list configs in ``_CARTESIAN_PROD_SIZES``
  (single / singleton / empty / equal-length / mixed-length / 3-4 inputs);
* the value-range sweep uses ``tu.make_input`` over every supported dtype, so
  every range round-trips exactly through the gather materialization;
* the multi-input analogue of broadcasting is combining inputs of unequal
  lengths (e.g. ``[1, 7]`` or ``[3, 1, 3]``), which the shape configs cover.

Every case below is one Workload (one pytest parametrization combo).
"""

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Value-range sweep + shape levels + backward + negative + nan/inf.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_INT8_DTYPES = [torch.int8, torch.uint8]

# Candidate dtypes: the spec's required 9 dtypes plus the wider float/int/bool
# sets; each one is probed for real support before it is parametrized.
_CANDIDATE_DTYPES = list(
    dict.fromkeys(
        _FP8_DTYPES
        + _INT8_DTYPES
        + utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + utils.BOOL_TYPES
    )
)


def _probe_dtype(operator, dtype):
    """Return whether the real aten op accepts ``dtype`` on this device.

    ``tu.supported_dtypes``'s default probe calls the op with a single tensor,
    which does not match the list-of-1-D-tensors schema, so pass a custom probe
    that builds two 1-D inputs. Any exception means "unsupported".
    """
    try:
        first = tu.make_input(dtype, (4,), ["-1", "1"])
        second = tu.make_input(dtype, (3,), ["-1", "1"])
        getattr(torch.ops.aten, operator).default([first, second])
    except Exception:
        return False
    return True


_SUPPORTED_DTYPES = tu.supported_dtypes(
    "cartesian_prod", candidates=_CANDIDATE_DTYPES, probe=_probe_dtype
)
if not _SUPPORTED_DTYPES:
    # Never collect zero dtype cases: fall back to the full candidate list so a
    # failed/absent probe never silently drops the spec-required int8/uint8/fp8
    # dtypes.
    _SUPPORTED_DTYPES = list(_CANDIDATE_DTYPES)

_FP8_DTYPE_SET = {torch.float8_e4m3fn, torch.float8_e5m2}

# fp32/bf16/fp16 are the autograd-capable float dtypes; fp8 has no autograd and
# fp64 support is device dependent, so only the shared float set is used there.
_FLOAT_DTYPES = [d for d in _SUPPORTED_DTYPES if d in utils.ALL_FLOAT_DTYPES]
_BACKWARD_DTYPES = [d for d in _FLOAT_DTYPES if d in utils.FLOAT_DTYPES]
_NAN_INF_DTYPES = [d for d in _FLOAT_DTYPES if d not in _FP8_DTYPE_SET]

# The value-range grid applies to every supported non-bool dtype; bool ignores
# the range (it is still covered, range-independently, by the shape grid).
_RANGE_DTYPES = [d for d in _SUPPORTED_DTYPES if d != torch.bool]


_RANGE_PAIRS = [(d, r) for d in _RANGE_DTYPES for r in tu.selected_ranges()]

# Shape levels: each entry is the list of 1-D input sizes (the op's shape
# dimension). The spec's seven dense shapes do not apply to this op: its input
# is ``Tensor[]`` (a list of 1-D tensors, one per operand), so the natural shape
# parameter is the number and lengths of those inputs rather than a single dense
# shape -- a lone tuple like (256,) cannot express "k operands of length n".
# These configs therefore cover single / singleton / empty / equal-length /
# mixed-length / 3-4 input cases and the resulting empty outputs.
_CARTESIAN_PROD_SIZES = (
    [[8], [3, 5], [2, 4, 3]]
    if tu.LEVEL == "quick"
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

# Backward input lists stay small (the autograd graph is built per input).
_BACKWARD_SIZES = (
    [[8], [3, 5]] if tu.LEVEL == "quick" else [[8], [3, 5], [2, 4, 3], [16, 16]]
)


def _resolve_gems_op():
    """Resolve the candidate inside the test (never at import time).

    Resolution order: (1) the process-local override installed by KernelGen,
    (2) the direct ``flag_gems.cartesian_prod`` callable, (3) ``None`` when
    neither exists yet (the tests then run against the PyTorch reference so the
    file stays runnable before an implementation is merged).
    """
    try:
        return flag_gems.testing.resolve_gems_op(
            "cartesian_prod", getattr(flag_gems, "cartesian_prod", None)
        )
    except LookupError:
        return None


def _apply_cartesian_prod(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        return torch.ops.aten.cartesian_prod(inp)
    return gems_op(inp)


def _assert_close(res_out, ref_out, dtype):
    """Floats use ``gems_assert_close``; int/bool/fp8 must be bit-exact.

    ``torch.testing.assert_close`` has no working tolerance path for fp8, and
    the op is a pure gather, so fp8 is compared exactly like int/bool.
    """
    if (dtype.is_floating_point or dtype.is_complex) and dtype not in _FP8_DTYPE_SET:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _expected_grads(grad_output, sizes):
    """Analytic gradient of a pure gather per input.

    Input element ``i`` appears in exactly one position along its own dim of the
    flattened combination index, so the gradient is ``grad_output[:, i]``
    viewed as the full size tuple and summed over every other dim. A single
    input is returned as-is, so its gradient is the grad output verbatim.
    """
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
@pytest.mark.parametrize("dtype,value_range", _RANGE_PAIRS)
def test_cartesian_prod(sizes, dtype, value_range):
    # Full grid: every supported dtype x every value range x every shape level.
    inp = [tu.make_input(dtype, (size,), value_range) for size in sizes]
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
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cartesian_prod_row_order(dtype):
    # Exact semantic check: rows follow itertools.product order (first input
    # varies slowest) and each input element is copied verbatim.
    a = torch.tensor([0, 1, 2], dtype=dtype, device=flag_gems.device)
    b = torch.tensor([10, 20], dtype=dtype, device=flag_gems.device)
    expected = torch.tensor(
        [[0, 10], [0, 20], [1, 10], [1, 20], [2, 10], [2, 20]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = [utils.to_reference(a), utils.to_reference(b)]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    res_out = _apply_cartesian_prod([a, b])

    assert res_out.shape == ref_out.shape == (6, 2)
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, utils.to_reference(expected))
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
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
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_cartesian_prod_nan_inf(dtype):
    # Pure gather: +inf/-inf/nan/+-0.0 pass through unchanged
    # (assert_result_close uses equal_nan=True on the float path).
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
@pytest.mark.parametrize("sizes", _BACKWARD_SIZES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_cartesian_prod_backward(sizes, dtype):
    out_shape = (sizes[0],) if len(sizes) == 1 else (math.prod(sizes), len(sizes))
    inp = [
        tu.make_input(dtype, (size,), ["-1", "1"]).requires_grad_() for size in sizes
    ]
    grad = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_inp = [utils.to_reference(t) for t in inp]
    ref_grad = utils.to_reference(grad)

    # The analytic gradient must match autograd on the aten reference...
    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)
    expected = _expected_grads(ref_grad, sizes)
    for got, exp in zip(ref_in_grads, expected):
        tu.assert_result_close(got, exp)

    # ...the candidate forward output must match the reference...
    res_out = _apply_cartesian_prod(inp)
    _assert_close(res_out, ref_out, dtype)

    # ...and, if the candidate output is differentiable, its gradient must match
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
@pytest.mark.parametrize("shape", [(3, 4), ()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_cartesian_prod_rejects_multidim_input(shape, dtype):
    # The op only accepts 1-D tensors; a 2-D or 0-dim input must raise.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
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
