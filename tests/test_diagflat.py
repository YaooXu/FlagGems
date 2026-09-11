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
count: the correctness shapes are therefore bounded (~1M output elements max)
and whole ranks rather than the generic multi-million-element levels are used.

Coverage follows the regular-operator spec adapted to a pure data-movement op:

* dtype coverage is probed with :func:`tu.supported_dtypes` (all of the spec's
  required dtypes -- int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/bf16/fp16/int32/int64
  -- plus bool are supported on the active backend here);
* shape levels: the spec's 7 shapes (0-D, which torch.diagflat accepts, through
  the large dense levels), bounded to inputs whose quadratic output stays small,
  plus the empty input;
* value ranges: the spec's five ranges via :func:`tu.make_input` (the values
  round-trip exactly through the diagonal placement);
* edge cases: empty inputs, large offsets (|offset| > numel), non-contiguous
  (transposed and strided) inputs and nan/inf/-inf passthrough;
* backward: ``autograd.grad`` validated against the analytic
  ``diag(grad_out, offset)`` gradient;
* negative: non-tensor input and non-int offset raise on both paths.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ---------------------------------------------------------------------------
# Dtype support (probe before writing cases, per the spec)
# ---------------------------------------------------------------------------
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_PROBE_DTYPES = [
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
# The probe builds a tiny input per dtype and calls the real aten op, treating
# any exception as "unsupported" (see tests/test_utils.py). If the probe yields
# nothing, keep the full candidate list rather than a float32-only fallback, so
# a failed/absent probe never silently drops the spec-required int8/uint8/fp8
# dtypes.
_DIAGFLAT_DTYPES = tu.supported_dtypes("diagflat", _PROBE_DTYPES)
if not _DIAGFLAT_DTYPES:
    _DIAGFLAT_DTYPES = list(_PROBE_DTYPES)

# Every floating dtype is exercised here, fp8 included. The narrow types do
# carry the special values: float8_e5m2 stores +/-inf and nan bit-for-bit and
# float8_e4m3fn preserves nan (its inf overflows to nan). The comparison below
# is done in fp32 because torch.testing.assert_close on an fp8 pair raises
# RuntimeError("mul_cpu_reduced_float" not implemented for 'Float8_e5m2') -- a
# CPU mul gap in the comparison, which fails even for two identical fp8
# tensors -- so it is the comparison, not the dtype, that has to be adapted.
_NAN_INF_DTYPES = [d for d in _DIAGFLAT_DTYPES if d.is_floating_point]
# double precision gives an exact analytic-gradient check; other float types
# only get the candidate-vs-reference check when autograd is available.
_GRAD_DTYPES = [d for d in _DIAGFLAT_DTYPES if d in (torch.float32, torch.float64)] or [
    torch.float32
]

_DIAGFLAT_OFFSETS = [-2, -1, 0, 1, 2]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _bounded_selected_shapes(limit=1024):
    """The spec shape levels whose quadratic diagflat output stays small enough."""
    return [shape for shape in tu.selected_shapes() if _numel(shape) <= limit]


# Shape levels aligned with the spec's 7 shapes (``tu.selected_shapes()``).
# diagflat accepts any input rank -- including 0-D, which torch.diagflat accepts
# and maps to a (1, 1) matrix -- so the spec shape set is used directly instead
# of a bespoke list. It is bounded to numel <= 1024 because the output side is
# ``numel(self) + |offset|``: the output element count is quadratic in the
# input, so the spec's multi-dim levels (``(1024, 1024)`` numel 1M,
# ``(20, 320, 15)`` numel 96K, ...) would allocate multi-gigabyte/terabyte
# outputs. Only the 0-D/1-D spec levels survive the bound, so small
# representative multi-dim shapes are added below to preserve rank coverage;
# ``(0,)`` (absent from the spec set) covers the empty-input case.
_DIAGFLAT_SHAPES = (
    _bounded_selected_shapes() + [(2, 3), (4, 5, 6), (2, 2, 2, 2, 3)] + [(0,)]
)

# Small inputs for the value-range sweep (bounded spec levels + rank reps).
_DIAGFLAT_RANGE_SHAPES = _bounded_selected_shapes() + [(2, 3), (4, 5, 6)]

_DIAGFLAT_NONCONTIG_SHAPES = [(4, 8), (6, 3), (2, 3, 4)]

_DIAGFLAT_STRIDED_SHAPES = [(16, 32), (4, 8, 16)]

_DIAGFLAT_BACKWARD_SHAPES = [(8,), (2, 3), (4, 5, 6)]


def _diagflat_shapes():
    """The bounded spec shape levels for the main sweep."""
    if tu.LEVEL == "quick":
        return [(2, 19, 7)]
    return list(_DIAGFLAT_SHAPES)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so the process-local
    # override installed by KernelGen for this run wins.
    return flag_gems.testing.resolve_gems_op(
        "diagflat", getattr(flag_gems, "diagflat", None)
    )


def _assert_output(res_out, ref_out, dtype):
    # diagflat materializes a new contiguous tensor (never an aliasing view):
    # shape, dtype, contiguity, view-ness and the diagonal placement must all
    # match the aten reference.
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    assert not res_out._is_view()
    if dtype in _FP8_DTYPES:
        # assert_close does not handle fp8 pairs directly; compare in fp32.
        tu.assert_result_close(res_out.float(), ref_out.float())
    elif dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _diagflat_shapes())
@pytest.mark.parametrize("offset", _DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    # Shape levels x offsets x every supported dtype with values in the default
    # [-1, 1] range (0-D, 1-D, empty, 2-D, 3-D, 4-D and 5-D are all covered).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_value_ranges(shape, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate ranges) must round-trip exactly
    # through the diagonal placement.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(2,), (16,)])
@pytest.mark.parametrize("offset", [-7, -3, 3, 7])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_large_offset(shape, offset, dtype):
    # Offsets whose magnitude may exceed the number of elements: the flattened
    # vector is placed on a diagonal that starts past the main diagonal,
    # leaving extra zero rows/columns around it.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


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
    ref_inp = utils.to_reference(inp)
    inp = inp.transpose(-1, -2)
    ref_inp = ref_inp.transpose(-1, -2)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_STRIDED_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_strided(shape, offset, dtype):
    # A strided slice (non-unit strides along the last dim) must be flattened
    # in logical view order too, so the candidate must read through the input's
    # actual strides. Slice on both devices so the layouts match.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_diagflat_nan_inf(dtype):
    # diagflat is a pure data-movement op: +inf/-inf/nan/+-0.0 pass through
    # unchanged onto the diagonal (assert_result_close uses equal_nan=True).
    # The values that reach the tensor are dtype-dependent: fp16/bf16 overflow
    # 1e30 to inf, and float8_e4m3fn turns every inf into nan, but both the
    # candidate and the reference are built from these same stored values, so
    # the comparison must hold whatever the dtype did to them. fp8 is compared
    # after casting to fp32 (see _NAN_INF_DTYPES).
    values = torch.tensor(
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
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.diagflat(ref_inp, 1)
    res_out = _resolve_gems_op()(values, 1)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if dtype in _FP8_DTYPES:
        tu.assert_result_close(res_out.float(), ref_out.float())
    else:
        tu.assert_result_close(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [-4, -1, 0, 1, 4])
@pytest.mark.parametrize("dtype", _DIAGFLAT_DTYPES)
def test_diagflat_empty_input(offset, dtype):
    # An empty input has no elements to place: offset 0 yields a 0x0 output and
    # |offset| > 0 yields an all-zero |offset| x |offset| matrix.
    inp = tu.make_input(dtype, (0,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", _DIAGFLAT_BACKWARD_SHAPES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test_diagflat_backward(shape, offset, dtype):
    # The forward op places flat_inp[k] at out[k, k+offset], so
    # d(diagflat(x))/dx extracts the offset-th diagonal of grad_output and
    # reshapes it back to the input shape (a pure gather, no arithmetic).
    # Validate the autograd reference against that analytic value, then check
    # the candidate forward output and -- only when the candidate output is
    # differentiable -- its gradient against the reference gradient.
    n = _numel(shape)
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, (n + abs(offset), n + abs(offset)), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    if dtype in (torch.float32, torch.float64):
        expected = torch.ops.aten.diag(ref_grad, offset).reshape(shape)
        tu.assert_result_close(ref_in_grad, expected)

    res_out = _resolve_gems_op()(inp, offset)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, ref_in_grad)


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
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.diagflat(ref_inp, 1.5)
    gems_op = _resolve_gems_op()
    with pytest.raises((TypeError, RuntimeError)):
        gems_op(inp, 1.5)
