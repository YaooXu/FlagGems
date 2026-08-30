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

# aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
# returns a 2-D tensor whose rows are all length-r combinations of the elements
# of a 1-D input (one combination per row). Without replacement there are
# C(n, r) rows and with replacement C(n + r - 1, r) rows; r=1 returns column
# vectors of shape (n, 1); r=0 yields a degenerate 1-D empty tensor; r > n
# without replacement yields an empty (0, r) result. The op is a pure gather of
# input elements (no arithmetic), so it works for every storage dtype, the
# output dtype always matches the input dtype and the input is never mutated.
#
# Coverage follows the regular-operator spec adapted to this 1-D gather op:
#   * shape levels: 1-D sizes selected by tu.LEVEL (quick/core/all), since the
#     generic multi-dim shape sets are not valid inputs for combinations;
#   * value ranges: tu.selected_ranges() over small 1-D inputs for every float
#     and int dtype (the op round-trips values bit-exactly, so the full
#     -1/0/1/max/min and degenerate constant ranges must pass through);
#   * edge cases: empty inputs, non-contiguous (strided) inputs, r boundaries
#     (0 and r > n), and nan/inf/-inf/+-0.0 passthrough;
#   * backward: the gradient is the reverse of the index gather (scatter the
#     grad_output rows back to the input positions); autograd.grad() on the
#     reference is validated against that analytic scatter and, when the
#     candidate output is differentiable, the candidate gradient is compared to
#     the reference gradient;
#   * negative: multi-dim (and 0-dim) inputs, a negative r and a non-int r all
#     raise on the aten reference and must raise on the candidate.
# Each pytest parametrization combo below is one Workload.
if tu.LEVEL == "quick":
    _COMBINATIONS_SHAPES = [(4,), (8,)]
elif tu.LEVEL in ("all", "extended"):
    # The largest all-level case (96, r=3) writes C(96, 3) * 3 = 428,640
    # output elements, staying under the 1M-element correctness cap.
    _COMBINATIONS_SHAPES = [(1,), (2,), (4,), (8,), (16,), (64,), (96,)]
else:  # core
    _COMBINATIONS_SHAPES = [(1,), (2,), (4,), (8,), (16,), (64,)]

# Small inputs for the value-range sweep: combinations is a pure gather, so two
# sizes are enough to exercise the full spec range list per dtype.
_COMBINATIONS_RANGE_SHAPES = [(8,), (16,)]

_COMBINATIONS_R = [1, 2, 3]

_COMBINATIONS_WITH_REPLACEMENT = [False, True]

_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_INT_DTYPES = utils.ALL_INT_DTYPES
_COMBINATIONS_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + utils.BOOL_TYPES


def _resolve_gems_op():
    # Resolution order: (1) the process-local override installed by KernelGen
    # via flag_gems.testing.override_gems_op, (2) the direct
    # flag_gems.combinations callable once it is registered, (3) None -> the
    # test falls back to the PyTorch reference so it stays runnable before a
    # FlagGems implementation is registered.
    try:
        return flag_gems.testing.resolve_gems_op(
            "combinations", getattr(flag_gems, "combinations", None)
        )
    except LookupError:
        return None


def _combinations_op():
    gems_op = _resolve_gems_op()
    if gems_op is None:
        return torch.ops.aten.combinations
    return gems_op


def _assert_match(res_out, ref_out, dtype):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _expected_combination_grad(n, r, with_replacement, grad_output):
    # Each output row is one r-combination of input indices, so the analytic
    # gradient scatters grad_output back to the input positions. index_add_
    # also accumulates the diagonal (i, i) rows of the with_replacement case
    # correctly (each occurrence of the index contributes once).
    idx = torch.ops.aten.combinations(
        torch.arange(n, dtype=torch.long, device=grad_output.device),
        r,
        with_replacement,
    )
    grad = torch.zeros(n, dtype=grad_output.dtype, device=grad_output.device)
    grad.index_add_(0, idx.reshape(-1), grad_output.reshape(-1))
    return grad


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _COMBINATIONS_SHAPES)
@pytest.mark.parametrize("r", _COMBINATIONS_R)
@pytest.mark.parametrize("with_replacement", _COMBINATIONS_WITH_REPLACEMENT)
@pytest.mark.parametrize("dtype", _COMBINATIONS_DTYPES)
def test_combinations(shape, r, with_replacement, dtype):
    # Shape levels x r x replacement mode x every supported dtype. Values come
    # from the default [-1, 1] range, so each dtype sees negative and positive
    # entries; the row count C(n, r) / C(n + r - 1, r) and the exact value
    # gather are both compared against the aten reference.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _COMBINATIONS_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES + _INT_DTYPES)
def test_combinations_value_ranges(shape, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and degenerate constant ranges) must round-trip
    # exactly through the gather materialization. bool ignores the range and is
    # covered by the shape-level test above.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(inp, 2, False)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [1, 2, 3])
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_combinations_empty_input(r, with_replacement, dtype):
    # An empty 1-D input has no elements to combine: aten returns an empty
    # (0, r) tensor of the input dtype for every r / with_replacement setting.
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [0, 5])
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_r_boundaries(r, with_replacement, dtype):
    # n=4 boundary cases: r=0 returns a degenerate 1-D empty (0,) tensor;
    # r=5 > n without replacement returns an empty (0, 5) tensor; with
    # replacement the output still has C(4 + 5 - 1, 5) = 56 rows.
    inp = tu.make_input(dtype, (4,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _COMBINATIONS_DTYPES)
def test_combinations_non_contiguous(dtype):
    # combinations gathers input elements by index, so the candidate must read
    # through the input's actual strides. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, (32,), ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(inp, 2, False)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_combinations_nan_inf(dtype):
    # combinations is a pure gather: +inf/-inf/nan/+-0.0 pass through unchanged
    # (equal_nan=True is active on the float path of assert_result_close; 1e30
    # overflows to inf in fp16/bf16 on both paths identically).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(values, 2, False)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", _COMBINATIONS_R)
@pytest.mark.parametrize("with_replacement", _COMBINATIONS_WITH_REPLACEMENT)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_combinations_backward(r, with_replacement, dtype):
    # The forward op is an index gather, so its gradient scatters grad_output
    # back to the input positions. Compute the reference gradient with
    # autograd.grad() on the CPU reference, validate it against the analytic
    # scatter (on fp32/fp64, where both algorithms round identically), then
    # check the candidate forward output and - only when the candidate output
    # is differentiable - its gradient against the reference gradient.
    n = 8
    rows = math.comb(n + r - 1, r) if with_replacement else math.comb(n, r)
    inp = tu.make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (rows, r), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    if dtype in (torch.float32, torch.float64):
        expected = _expected_combination_grad(n, r, with_replacement, ref_grad)
        tu.assert_result_close(ref_in_grad, expected)

    res_out = _combinations_op()(inp, r, with_replacement)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", [(), (4, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_raises_on_non_1d(shape, dtype):
    # aten::combinations only accepts 1-D inputs (0-dim scalars and 2-D+ tensors
    # are rejected); the candidate must raise the same way.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2, False)
    gems_op = _combinations_op()
    with pytest.raises(RuntimeError):
        gems_op(inp, 2, False)


@pytest.mark.combinations
def test_combinations_raises_on_negative_r():
    # r must be non-negative; aten raises RuntimeError and the candidate must
    # behave the same way.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, -1, False)
    gems_op = _combinations_op()
    with pytest.raises(RuntimeError):
        gems_op(inp, -1, False)


@pytest.mark.combinations
def test_combinations_raises_on_non_int_r():
    # The schema demands an int r; passing a float must raise on both paths.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2.0, False)
    gems_op = _combinations_op()
    with pytest.raises((TypeError, RuntimeError)):
        gems_op(inp, 2.0, False)
