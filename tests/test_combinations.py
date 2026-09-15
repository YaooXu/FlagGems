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
#     bf16/fp16/int32/int64) plus fp64/int16/bool;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over the 1-D entries of tu.selected_shapes() (the shared set is
#     multi-dim, and combinations only accepts 1-D inputs) for every declared
#     dtype, so each dtype x range pair is one Workload;
#   * shapes -- level-driven 1-D sizes crossed with r and with_replacement,
#     including the r == 0 and r > n boundaries and the empty input;
#   * edge cases -- non-contiguous (strided) inputs, nan/inf/-inf/+-0.0
#     passthrough, and a no-mutation check;
#   * backward -- the forward op is an index gather, so its gradient scatters
#     grad_output back to the input positions; the candidate gradient is
#     compared with autograd.grad() on the ATen reference;
#   * negative -- multi-dim (and 0-dim) inputs, a negative r and a non-int r all
#     raise on the aten reference and must raise on the candidate.
# Broadcast does not apply: the op is unary and takes a single labelled tensor.
# Each pytest parametrization combo below is one Workload.

# Combinations gathers stored values, including both FP8 formats.
_DTYPES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)


_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]
# FP8 nonempty backward needs masked_scatter/add kernels absent on CPU/CUDA.
# The regular gradient grid uses dtypes supported by both reference modes.
_BACKWARD_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point and dtype not in _FP8_DTYPES
]

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
_MUTATION_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.float16, torch.int32, torch.bool)
    if dtype in _DTYPES
] or _DTYPES[:1]

# combinations is a 1-D op, so the shared spec shape set is reduced to its 1-D
# entries. The quick level has no 1-D entry in tu.selected_shapes(), so a small
# local fallback keeps the grid populated at both levels.
_SPEC_1D_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) == 1] or [
    (4,),
    (8,),
]

# Representative 1-D sizes for the r/replacement sweep. At n=96 and r=3,
# the no-replacement case produces C(96,3) * 3 = 428,640 output elements.
if tu.QUICK_MODE:
    _LEVEL_SHAPES = [(4,), (8,)]
else:
    _LEVEL_SHAPES = [(1,), (2,), (4,), (8,), (16,), (64,), (96,)]

_R_VALUES = [1, 2, 3]
_REPLACEMENT_MODES = [False, True]


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "combinations", getattr(flag_gems, "combinations", None)
    )


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _SPEC_1D_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_spec_shapes_value_ranges(shape, value_range, dtype):
    # The 1-D entries of the shared spec shape set crossed with the five spec
    # value ranges for every declared dtype. The op never transforms the stored
    # values, so the full range sweep (including 0/max/min and degenerate
    # constant ranges) must round-trip exactly through the gather
    # materialization. bool ignores the range and is covered here as well.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _resolve_gems_op()(inp, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _LEVEL_SHAPES)
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_combinations_shapes_r_replacement(shape, r, with_replacement, dtype):
    # Level-driven 1-D sizes x r x replacement mode. Values come from the
    # default [-1, 1] range; the row count C(n, r) / C(n + r - 1, r) and the
    # exact value gather are both compared against the aten reference.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _resolve_gems_op()(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", _EMPTY_DTYPES)
def test_combinations_empty_input(r, with_replacement, dtype):
    # An empty 1-D input has no elements to combine: aten returns an empty
    # (0, r) tensor of the input dtype for every r / with_replacement setting
    # (and a 1-D (0,) tensor when r == 0).
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _resolve_gems_op()(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [0, 5, 10])
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_r_boundaries(r, with_replacement, dtype):
    # n=4 boundary cases: r=0 returns a degenerate 1-D empty (0,) tensor;
    # r=5/10 > n without replacement returns an empty (0, r) tensor; with
    # replacement the output still has C(n + r - 1, r) rows.
    inp = tu.make_input(dtype, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _resolve_gems_op()(inp, r, with_replacement)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _DTYPES)
def test_combinations_non_contiguous(dtype):
    # combinations gathers input elements by index, so the candidate must read
    # through the input's actual strides. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, (32,), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _resolve_gems_op()(inp, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DTYPES))
)
def test_combinations_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _resolve_gems_op()(values, 2, False)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _MUTATION_DTYPES)
def test_combinations_does_not_mutate_input(dtype):
    # The op only materializes a gather; the source tensor must be untouched.
    inp = tu.make_input(dtype, (16,), ["-1", "1"])
    before = tu.to_reference(inp)

    _resolve_gems_op()(inp, 2, False)

    tu.assert_result_equal(inp, before)


@pytest.mark.combinations
@pytest.mark.parametrize("n", [0, 1, 8])
@pytest.mark.parametrize("r", _R_VALUES)
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES))
def test_combinations_backward(n, r, with_replacement, dtype):
    # Forward is an exact gather; backward sums contributions for each input.
    inp = tu.make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    grad = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _resolve_gems_op()(inp, r, with_replacement)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    # Singleton gathers copy gradients; empty results contribute exact zeros.
    if r == 1 or n == 0 or (r > n and not with_replacement):
        tu.assert_result_equal(res_in_grad, ref_in_grad)
    else:
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.combinations
@pytest.mark.parametrize("n", [0, 1, 8])
@pytest.mark.parametrize("with_replacement", _REPLACEMENT_MODES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_FLOAT_DTYPES))
def test_combinations_zero_r_no_autograd(n, with_replacement, dtype):
    # r=0 returns a fresh empty result with no gradient connection to the input.
    inp = tu.make_input(dtype, (n,), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.combinations(ref_inp, 0, with_replacement)
    res_out = _resolve_gems_op()(inp, 0, with_replacement)
    tu.assert_result_equal(res_out, ref_out)
    assert not res_out.requires_grad
    assert res_out.grad_fn is None


@pytest.mark.combinations
@pytest.mark.parametrize("shape", [(), (4, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_raises_on_non_1d(shape, dtype):
    # aten::combinations only accepts 1-D inputs (0-dim scalars and 2-D+ tensors
    # are rejected); the candidate must raise the same way.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2, False)
    gems_op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        gems_op(inp, 2, False)


@pytest.mark.combinations
def test_combinations_raises_on_negative_r():
    # r must be non-negative; aten raises RuntimeError and the candidate must
    # behave the same way.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, -1, False)
    gems_op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        gems_op(inp, -1, False)


@pytest.mark.combinations
def test_combinations_raises_on_non_int_r():
    # The schema demands an int r; passing a float must raise on both paths.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2.0, False)
    gems_op = _resolve_gems_op()
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        gems_op(inp, 2.0, False)
