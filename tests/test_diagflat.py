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

"""Correctness tests for ``aten::diagflat(Tensor self, int offset=0) -> Tensor``.

``diagflat`` flattens ``self`` (in logical row-major view order) into a 1-D
vector and returns a NEW 2-D square matrix whose ``offset``-th diagonal holds
that vector, with zeros everywhere else. The output side length is
``numel(self) + |offset|``, so the output is quadratic in the input element
count. The large spec shapes therefore use explicit smaller representatives
of the same rank; for example, a (1024,1024) input alone needs a 4 TiB float32
output at offset=0. This is an operator-specific allocation constraint.

Coverage follows the regular-operator spec adapted to a pure data-movement op:

* dtype coverage explicitly includes all of the spec's
  required dtypes -- int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/bf16/fp16/int32/int64
  -- plus float64/int16/bool;
* shape levels: scalar, singleton and regular 1-D spec shapes, explicit
  representatives for ranks 2 through 5, and the empty input;
* value ranges: the spec's five ranges via :func:`tu.make_input` (the values
  round-trip exactly through the diagonal placement);
* edge cases: empty inputs, large offsets (|offset| > numel), non-contiguous
  (transposed and strided) inputs and nan/inf/-inf passthrough;
* backward: candidate gradients compared exactly with ATen autograd;
* negative: non-tensor input and non-int offset raise on both paths.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------

_DIAGFLAT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.float64,
    torch.bool,
]

_GRAD_DTYPES = [d for d in _DIAGFLAT_DTYPES if d.is_floating_point]

_DIAGFLAT_OFFSETS = [-2, -1, 0, 1, 2]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


# Keep the scalar and 1-D spec shapes. Ranks 2-5 use explicit representatives:
# the original large shapes would require tens of GiB to hundreds of TiB for
# each float32 output, before allocating the reference and comparison buffers.
_DIAGFLAT_SHAPES = [
    (),
    (1,),
    (256,),
    (2, 3),
    (4, 5, 6),
    (2, 3, 4, 5),
    (2, 2, 2, 2, 3),
    (0,),
]

_DIAGFLAT_RANGE_SHAPES = tu.selected_cases(
    _DIAGFLAT_SHAPES[:-1], quick=[(2, 19, 7), (2, 3), (4, 5, 6)]
)

_DIAGFLAT_NONCONTIG_SHAPES = [(4, 8), (6, 3), (2, 3, 4)]

_DIAGFLAT_STRIDED_SHAPES = [(16, 32), (4, 8, 16)]

_DIAGFLAT_BACKWARD_SHAPES = [(8,), (2, 3), (4, 5, 6)]


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "diagflat", getattr(flag_gems, "diagflat", None)
    )


def _assert_output(res_out, ref_out):
    # diagflat materializes a new contiguous tensor (never an aliasing view):
    # shape, dtype, contiguity, view-ness and the diagonal placement must all
    # match the aten reference.
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize(
    "shape", tu.selected_cases(_DIAGFLAT_SHAPES, quick=[(2, 19, 7)])
)
@pytest.mark.parametrize("offset", _DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    # Shape levels x offsets x every supported dtype with values in the default
    # [-1, 1] range (0-D, 1-D, empty, 2-D, 3-D, 4-D and 5-D are all covered).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_value_ranges(shape, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate ranges) must round-trip exactly
    # through the diagonal placement.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(2,), (16,)])
@pytest.mark.parametrize("offset", [-7, -3, 3, 7])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_large_offset(shape, offset, dtype):
    # Offsets whose magnitude may exceed the number of elements: the flattened
    # vector is placed on a diagonal that starts past the main diagonal,
    # leaving extra zero rows/columns around it.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_NONCONTIG_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_non_contiguous(shape, offset, dtype):
    # diagflat flattens the logical view, so a transposed (non-contiguous)
    # input must produce a different diagonal order than a contiguous one.
    # Transpose on both the test device and the reference device so the two
    # inputs share the same memory layout.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    inp = inp.transpose(-1, -2)
    ref_inp = ref_inp.transpose(-1, -2)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_STRIDED_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_strided(shape, offset, dtype):
    # A strided slice (non-unit strides along the last dim) must be flattened
    # in logical view order too, so the candidate must read through the input's
    # actual strides. Slice on both devices so the layouts match.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DIAGFLAT_DTYPES))
)
def test_diagflat_nan_inf(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.diagflat(ref_inp, 1)
    res_out = _resolve_gems_op()(values, 1)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [-4, -1, 0, 1, 4])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_empty_input(offset, dtype):
    # An empty input has no elements to place: offset 0 yields a 0x0 output and
    # |offset| > 0 yields an all-zero |offset| x |offset| matrix.
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_BACKWARD_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", tu.selected_cases(_GRAD_DTYPES))
def test_diagflat_backward(shape, offset, dtype):
    # The forward op places flat_inp[k] at out[k, k+offset], so
    # d(diagflat(x))/dx extracts the offset-th diagonal of grad_output and
    # reshapes it back to the input shape (a pure gather, no arithmetic).
    n = _numel(shape)
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (n + abs(offset), n + abs(offset)), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _resolve_gems_op()(inp, offset)
    tu.assert_result_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.diagflat
def test_diagflat_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(3.14)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gems_op(3.14)


@pytest.mark.diagflat
def test_diagflat_rejects_non_int_offset():
    # The schema demands an int offset; passing a float must raise on both
    # paths.
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(ref_inp, 1.5)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, RuntimeError)):
        gems_op(inp, 1.5)
