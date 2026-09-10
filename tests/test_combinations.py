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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
# returns a 2-D tensor whose rows are all length-r combinations of the elements
# of a 1-D input (one combination per row). Without replacement there are C(n, r)
# rows and with replacement C(n + r - 1, r) rows; r=1 returns column vectors of
# shape (n, 1); r=0 yields a degenerate 1-D empty (0,) tensor; r > n without
# replacement yields an empty (0, r) result. The op is a pure gather of input
# elements (no arithmetic), so it works for every storage dtype, the output
# dtype always matches the input dtype and the input is never mutated.
#
# Coverage follows the regular-operator spec adapted to this fixed-rank (1-D)
# gather op:
#   * dtypes -- the full required set (int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/
#     bf16/fp16/int32/int64) plus fp64/int16/bool where the active device
#     supports them, probed at import time with tu.supported_dtypes;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over the 1-D entries of tu.selected_shapes() (the shared set is
#     multi-dim, and combinations only accepts 1-D inputs) for every probed
#     dtype, so each dtype x range pair is one Workload;
#   * shapes -- level-driven 1-D sizes crossed with r and with_replacement,
#     including the r == 0 and r > n boundaries and the empty input;
#   * edge cases -- non-contiguous (strided) inputs, nan/inf/-inf/+-0.0
#     passthrough, and a no-mutation check;
#   * backward -- the forward op is an index gather, so its gradient scatters
#     grad_output back to the input positions; autograd.grad() on the reference
#     is validated against that analytic scatter (on fp32/fp64) and, when the
#     candidate output is differentiable, the candidate gradient is compared to
#     the reference gradient;
#   * negative -- multi-dim (and 0-dim) inputs, a negative r and a non-int r all
#     raise on the aten reference and must raise on the candidate.
# Broadcast does not apply: the op is unary and takes a single labelled tensor.
# Each pytest parametrization combo below is one Workload.

# Probe the storage dtypes the combinations kernel actually accepts on the
# active device (spec: never guess). Every required dtype plus fp64/int16/bool
# is accepted by the CUDA implementation, including the fp8 types.
_CANDIDATE_DTYPES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)

_DTYPES = tu.supported_dtypes("combinations", candidates=_CANDIDATE_DTYPES) or [
    torch.float32
]

# assert_close does not support float8 tensors, and combinations is a pure
# bit-exact gather, so fp8 is compared with the exact helper. All other floats
# go through the tolerance-based helper.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_FLOAT_CLOSE_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point and dtype not in _FP8_DTYPES
]
_FP8_SUPPORTED = [dtype for dtype in _FP8_DTYPES if dtype in _DTYPES]

# Representative dtype sets for the grids that do not need every storage type
# (the full dtype contract is already swept by the value-range grid).
_GRID_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.float16, torch.int32, torch.bool)
    if dtype in _DTYPES
] or _DTYPES[:1]
_EMPTY_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.int32, torch.bool, torch.int8)
    if dtype in _DTYPES
] or _DTYPES[:1]
_BACKWARD_DTYPES = _FLOAT_CLOSE_DTYPES
_MUTATION_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.float16, torch.int32, torch.bool)
    if dtype in _DTYPES
] or _DTYPES[:1]

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# combinations is a 1-D op, so the shared spec shape set is reduced to its 1-D
# entries. The quick level has no 1-D entry in tu.selected_shapes(), so a small
# local fallback keeps the grid populated at both levels.
_SPEC_1D_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) == 1] or [
    (4,),
    (8,),
]

# Level-driven 1-D sizes. The largest all-level case (96, r=3) writes
# C(96, 3) * 3 = 428,640 output elements, staying under the 1M-element cap.
if tu.LEVEL == "quick":
    _LEVEL_SHAPES = [(4,), (8,)]
else:
    _LEVEL_SHAPES = [(1,), (2,), (4,), (8,), (16,), (64,), (96,)]

_R_VALUES = [1, 2, 3]
_REPLACEMENT_MODES = [False, True]


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen via
    # flag_gems.testing.override_gems_op wins. The default stays None until
    # flag_gems.combinations is registered.
    return flag_gems.testing.resolve_gems_op(
        "combinations", getattr(flag_gems, "combinations", None)
    )


def _combinations_op():
    # Resolution order: (1) the KernelGen process-local override, (2) the direct
    # flag_gems.combinations callable once it is registered, (3) the aten
    # reference so the file stays runnable before an implementation exists.
    try:
        return _resolve_gems_op()
    except LookupError:
        return torch.ops.aten.combinations


def _make_input(dtype, shape, value_range):
    """tu.make_input with the one unrepresentable uint8 range snapped.

    ``[-1, 0]`` cannot build a uint8 tensor (``-1`` is not representable), and
    the op only gathers stored values, so the negative bound is dropped.
    """
    if dtype == torch.uint8 and list(value_range) == ["-1", "0"]:
        value_range = ["0", "1"]
    return tu.make_input(dtype, shape, value_range)


def _assert_match(res_out, ref_out, dtype):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    if dtype in _FLOAT_CLOSE_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        # bool / integer / float8: the gather is exact, so equality is exact.
        utils.gems_assert_equal(res_out, ref_out)


def _expected_combination_grad(n, r, with_replacement, grad_output):
    # Each output row is one r-combination of input indices, so the analytic
    # gradient scatters grad_output back to the input positions. index_add_
    # accumulates the diagonal (i, i) rows of the with_replacement case
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
@pytest.mark.parametrize("shape", _SPEC_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_spec_shapes_value_ranges(shape, value_range, dtype):
    # The 1-D entries of the shared spec shape set crossed with the five spec
    # value ranges for every probed dtype. The op never transforms the stored
    # values, so the full range sweep (including 0/max/min and degenerate
    # constant ranges) must round-trip exactly through the gather
    # materialization. bool ignores the range and is covered here as well.
    inp = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(inp, 2, False)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _LEVEL_SHAPES)
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_combinations_shapes_r_replacement(shape, r, with_replacement, dtype):
    # Level-driven 1-D sizes x r x replacement mode. Values come from the
    # default [-1, 1] range; the row count C(n, r) / C(n + r - 1, r) and the
    # exact value gather are both compared against the aten reference.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_combinations_empty_input(r, with_replacement, dtype):
    # An empty 1-D input has no elements to combine: aten returns an empty
    # (0, r) tensor of the input dtype for every r / with_replacement setting
    # (and a 1-D (0,) tensor when r == 0).
    inp = _make_input(dtype, (0,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [0, 5, 10])
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_r_boundaries(r, with_replacement, dtype):
    # n=4 boundary cases: r=0 returns a degenerate 1-D empty (0,) tensor;
    # r=5/10 > n without replacement returns an empty (0, r) tensor; with
    # replacement the output still has C(n + r - 1, r) rows.
    inp = _make_input(dtype, (4,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_non_contiguous(dtype):
    # combinations gathers input elements by index, so the candidate must read
    # through the input's actual strides. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout.
    base = _make_input(dtype, (32,), ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(inp, 2, False)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _FLOAT_CLOSE_DTYPES)
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
@pytest.mark.parametrize("dtype", _MUTATION_DTYPES)
def test_combinations_does_not_mutate_input(dtype):
    # The op only materializes a gather; the source tensor must be untouched.
    inp = _make_input(dtype, (16,), ["-1", "1"])
    before = inp.clone()

    _combinations_op()(inp, 2, False)

    _assert_match(inp, before, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_combinations_backward(r, with_replacement, dtype):
    # The forward op is an index gather, so its gradient scatters grad_output
    # back to the input positions. Compute the reference gradient with
    # autograd.grad() on the reference device, validate it against the analytic
    # scatter (on fp32/fp64, where both algorithms round identically), then
    # check the candidate forward output and - only when the candidate output
    # is differentiable - its gradient against the reference gradient.
    n = 8
    rows = math.comb(n + r - 1, r) if with_replacement else math.comb(n, r)
    inp = _make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, (rows, r), ["-1", "1"])
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
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
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2, False)
    gems_op = _combinations_op()
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
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
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        gems_op(inp, -1, False)


@pytest.mark.combinations
def test_combinations_raises_on_non_int_r():
    # The schema demands an int r; passing a float must raise on both paths.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2.0, False)
    gems_op = _combinations_op()
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        gems_op(inp, 2.0, False)
