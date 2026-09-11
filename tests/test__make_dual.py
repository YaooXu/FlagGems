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
from torch.autograd.forward_ad import dual_level

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``_make_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._make_dual`` and ``-m _make_dual``
# both work.
setattr(
    pytest.mark,
    "_make_dual",
    MarkDecorator(Mark("_make_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
# is the forward-mode AD dual-construction primitive: it attaches ``tangent``
# to an aliasing view of ``primal`` at the forward-mode AD level ``level``. The
# level must already be active, so the native implementation is only callable
# inside ``torch.autograd.forward_ad.dual_level()``; the caller passes the
# level that the context assigned (its exact value varies across torch
# versions, so it is always taken from the context here). The observable value
# of the result is exactly ``primal`` (no arithmetic happens), so the candidate
# must reproduce primal's shape, dtype and storage layout, must alias primal
# (Tensor(a)), and must preserve the tangent: unpacking the result with
# ``torch.autograd.forward_ad.unpack_dual`` must recover primal and the
# original tangent unchanged. The primal must be floating-point or complex
# (aten raises on int/bool primals) and the tangent must match the primal's
# size; both constraints are respected below.
#
# Coverage follows the regular-operator spec adapted to a metadata/view op:
#   * shape levels: tu.selected_shapes() (ranks 0-5, selected by --quick);
#   * value ranges: tu.selected_ranges() over representative shapes, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the aliasing view round-trips all of them bit-for-bit,
#     and the tangent is preserved unchanged);
#   * dtype coverage: the probe-verified required dtypes (int8, uint8,
#     float8_e4m3fn, float8_e5m2, fp32, bf16, fp16, int32, int64) plus
#     float64/complex64 where the backend supports them. The dual path only
#     accepts floating-point/complex primals, so the int/bool dtypes are
#     covered as negative cases (rejection) instead;
#   * edge cases: non-contiguous (strided) primals, empty tensors, mutation
#     through the returned alias, and nan/inf/±0.0 special values;
#   * negative: int/bool primal, non-tensor primal, tangent/primal size
#     mismatch, non-int level and an inactive level are all rejected.
#
# No broadcast/backward dimensions apply: the tangent must match the primal's
# size exactly (aten rejects any broadcast shape), and _make_dual is a
# forward-AD construction primitive with no backward defined.

# fp8 primals are supported by aten and are a hard spec requirement, but
# torch.testing.assert_close cannot compare fp8 tensors on every build, so fp8
# comparisons upcast to float32 first (an exact, lossless widening).
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _unique(items):
    return list(dict.fromkeys(items))


def _dual_dtype_supported(dtype):
    """Probe the forward-AD dual path for ``dtype`` on the active device."""
    try:
        primal = torch.zeros((2,), dtype=dtype, device=flag_gems.device)
        tangent = torch.zeros((2,), dtype=dtype, device=flag_gems.device)
        with dual_level() as level:
            dual = torch.ops.aten._make_dual(primal, tangent, level)
            primal_out, tangent_out = torch.autograd.forward_ad.unpack_dual(dual)
        return (
            tangent_out is not None
            and tangent_out.dtype == dtype
            and primal_out.dtype == dtype
        )
    except Exception:
        return False


# Dtypes required by the operator-test spec. int8/uint8/int32/int64 are probed
# out because a forward tangent only exists for floating-point/complex storage.
_REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

# complex32 is excluded because the comparison helpers cannot materialize
# ComplexHalf; complex64 covers the complex branch exactly.
DUAL_DTYPES = [
    d
    for d in _unique(_REQUIRED_DTYPES + utils.ALL_FLOAT_DTYPES + [torch.complex64])
    if _dual_dtype_supported(d)
]

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_MAKE_DUAL_RANGE_SHAPES = [(), (256,), (7, 13, 29)]

_MAKE_DUAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MAKE_DUAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_MAKE_DUAL_EMPTY_SHAPES = [(0,), (2, 0, 3)]

# Levels probed on a plain tensor with no active dual_level(): every index,
# including 0, is inactive outside the context and must be rejected.
_INACTIVE_LEVELS = [-1, 0, 1, 3]

_MISMATCHED_SHAPES = [((4, 5), (3, 7)), ((4, 5), (5,)), ((16,), (8,))]

_NON_FLOAT_PRIMAL_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
]

_SPECIAL_DTYPES = _unique(utils.ALL_FLOAT_DTYPES + list(_FP8_DTYPES))


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._make_dual is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._make_dual callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_make_dual", getattr(flag_gems, "_make_dual", None)
    )


def _make_input(dtype, shape, value_range):
    # Unsigned dtypes cannot represent the negative low bound of the spec
    # ranges (e.g. uint8 with ["-1", "0"]): clamp the symbol to the dtype
    # minimum so make_input never builds an empty range. bool ignores ranges.
    low_symbol, high_symbol = value_range
    low_bound, _ = tu.dtype_bounds(dtype)
    if tu.resolve_bound(low_symbol, dtype) < low_bound:
        low_symbol = "0"
    return tu.make_input(dtype, shape, [low_symbol, high_symbol])


def _assert_close(res_out, ref_out, dtype):
    if dtype in _FP8_DTYPES:
        utils.gems_assert_equal(res_out.to(torch.float32), ref_out.to(torch.float32))
    elif dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_values(res_out, ref_out):
    # Value-range comparisons use the value-range-friendly tolerance helper,
    # except for fp8 which is upcast to float32 first (see above).
    if res_out.dtype in _FP8_DTYPES:
        utils.gems_assert_equal(res_out.to(torch.float32), ref_out.to(torch.float32))
    else:
        tu.assert_result_close(res_out, ref_out)


def _assert_exact_equal(res_out, ref_out, dtype):
    if dtype in _FP8_DTYPES:
        utils.gems_assert_equal(
            res_out.to(torch.float32), ref_out.to(torch.float32), equal_nan=True
        )
    else:
        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


def _assert_view_semantics(res_out, ref_out, inp):
    # _make_dual returns an aliasing view (Tensor(a)) of the primal: the
    # observable layout must match aten exactly and the result must share
    # storage with the candidate-side primal.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()


def _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype):
    # The whole purpose of the op is to produce a dual tensor: unpacking the
    # result must recover the primal value and the unchanged input tangent.
    # (unpack_dual of a plain non-dual tensor returns a None tangent, so the
    # candidate cannot skip the dual wrapping.)
    res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
    ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
    assert isinstance(res_tangent, torch.Tensor)
    assert isinstance(ref_tangent_out, torch.Tensor)
    _assert_close(res_primal, ref_primal, dtype)
    _assert_close(res_tangent, ref_tangent_out, dtype)
    _assert_close(res_tangent, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = _make_input(dtype, shape, ["-1", "1"])
    tangent = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_close(res_out, ref_out, dtype)
        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip bit-for-bit and the tangent must survive unchanged.
    inp = _make_input(dtype, shape, value_range)
    tangent = _make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
        ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
        _assert_values(res_primal, ref_primal)
        _assert_values(res_tangent, ref_tangent_out)
        _assert_values(res_tangent, ref_tangent)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_non_contiguous(shape, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous primal. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    tangent = _make_input(dtype, inp.shape, ["-1", "1"])
    ref_tangent = utils.to_reference(tangent)
    assert not inp.is_contiguous()

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_close(res_out, ref_out, dtype)
        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", DUAL_DTYPES)
def test__make_dual_empty(shape, dtype):
    # Empty tensors (0 elements) still carry a valid layout: the aliasing view
    # must preserve shape, strides, storage offset/data_ptr exactly and the
    # (also empty) tangent must round-trip.
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_close(res_out, ref_out, dtype)
        _assert_view_semantics(res_out, ref_out, inp)
        _assert_dual_semantics(res_out, ref_out, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__make_dual_mutation(shape, dtype):
    # The result is a true alias of the primal (Tensor(a)): writing through the
    # returned view must be observable on the candidate-side input, and the
    # reference must behave identically. The reference runs on an independent
    # clone so the two aliases are validated separately. The op itself never
    # mutates the primal or the tangent.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    tangent = _make_input(dtype, shape, ["-1", "1"])
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        ref_out.fill_(2.5)
        res_out.fill_(2.5)

        _assert_close(res_out, ref_out, dtype)
        assert res_out.data_ptr() == inp.data_ptr()
        tu.assert_result_close(inp, ref_inp)
        tu.assert_result_close(tangent, ref_tangent)


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", _SPECIAL_DTYPES)
def test__make_dual_special_values(dtype):
    # A pure alias must preserve every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(values, tangent, level)

        _assert_exact_equal(res_out, ref_out, dtype)
        # signbit has no fp8 kernel, so the sign check goes through float32.
        res_f = res_out.to(torch.float32)
        values_f = values.to(torch.float32)
        assert torch.signbit(res_f[0]).item() == torch.signbit(values_f[0]).item()
        assert torch.signbit(res_f[1]).item() == torch.signbit(values_f[1]).item()


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", _NON_FLOAT_PRIMAL_DTYPES)
def test__make_dual_rejects_non_float_primal(dtype):
    # Forward-mode dual tensors only support floating-point/complex storage:
    # aten raises on int/bool primals (the internal assert requires both primal
    # and tangent to be floating point or complex) and the candidate must too.
    with dual_level() as level:
        inp = _make_input(dtype, (4, 5), ["-1", "1"])
        tangent = _make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                utils.to_reference(inp), utils.to_reference(tangent), level
            )
        # The generated wrapper may fail on the first touch of the input
        # (attribute lookup, triton input validation or a dispatcher cast), so
        # accept the plausible Python failure modes; the point is that it must
        # fail rather than silently accept the int/bool primal.
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(inp, tangent, level)


@pytest.mark._make_dual
def test__make_dual_rejects_non_tensor_primal():
    # The aten schema requires a Tensor primal; a Python float hits the invalid
    # argument path and raises. The candidate must fail too rather than
    # silently accept scalars.
    with dual_level() as level:
        tangent = _make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(3.14, utils.to_reference(tangent), level)
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(3.14, tangent, level)


@pytest.mark._make_dual
@pytest.mark.parametrize("primal_shape,tangent_shape", _MISMATCHED_SHAPES)
def test__make_dual_rejects_tangent_size_mismatch(primal_shape, tangent_shape):
    # The tangent must have exactly the primal's size: aten rejects any other
    # shape (broadcasting is not defined for forward tangents) and the
    # candidate must reproduce the validation.
    with dual_level() as level:
        inp = _make_input(torch.float32, primal_shape, ["-1", "1"])
        tangent = _make_input(torch.float32, tangent_shape, ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                utils.to_reference(inp), utils.to_reference(tangent), level
            )
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(inp, tangent, level)


@pytest.mark._make_dual
@pytest.mark.parametrize("level", _INACTIVE_LEVELS)
def test__make_dual_rejects_inactive_level(level):
    # The named level must be live: outside dual_level() aten rejects any level
    # index with RuntimeError and the candidate must reproduce the validation
    # instead of silently ignoring the level.
    inp = _make_input(torch.float32, (4, 5), ["-1", "1"])
    tangent = _make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual(
            utils.to_reference(inp), utils.to_reference(tangent), level
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, tangent, level)


@pytest.mark._make_dual
def test__make_dual_rejects_non_int_level():
    # ``level`` is an int in the schema; a float is a cast error at the
    # dispatcher boundary and must be rejected by the candidate as well. The
    # dual_level() context is entered so an active level exists and the schema
    # cast is the only thing under test.
    with dual_level():
        inp = _make_input(torch.float32, (4, 5), ["-1", "1"])
        tangent = _make_input(torch.float32, (4, 5), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                utils.to_reference(inp), utils.to_reference(tangent), 1.5
            )
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(inp, tangent, 1.5)
