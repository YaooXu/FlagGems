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

# The KernelGen verification harness stages these files in an isolated copy
# of the FlagGems tree whose parent directory is not on sys.path. Make the
# ``tests``/``benchmark`` packages importable regardless of the harness
# process's sys.path so the relative imports below resolve.
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402
from torch.autograd.forward_ad import dual_level  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_unpack_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._unpack_dual`` and ``-m _unpack_dual``
# both work.
setattr(
    pytest.mark,
    "_unpack_dual",
    MarkDecorator(Mark("_unpack_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor
# tangent) is the forward-mode AD dual-construction inverse: given a dual tensor
# created at an active forward-mode AD ``level`` (e.g. by
# ``torch.ops.aten._make_dual``), it returns the primal as an aliasing view
# (Tensor(a), same shape, strides, storage offset, dtype and data_ptr) and the
# tangent registered at that level. The level must already be active, so the
# native implementation is only callable inside
# ``torch.autograd.forward_ad.dual_level()``; the caller passes the level that
# the context assigned (its exact value varies across torch versions, so it is
# always taken from the context here). On a plain tensor with no forward tangent
# at ``level`` the tangent component is None while the primal view is still
# returned. No arithmetic happens, so the op is pure metadata/view manipulation:
# primal values, tangent values and the alias contract are all asserted below.
#
# Coverage follows the regular-operator spec adapted to a metadata/view op:
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by TEST_LEVEL);
#   * value ranges: tu.selected_ranges() over representative shapes, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the aliasing view round-trips all of them bit-for-bit,
#     and the tangent is preserved unchanged);
#   * edge cases: non-contiguous (strided) duals, mutation through the returned
#     primal alias, empty tensors, and nan/inf/±0.0 special values;
#   * negative: non-tensor input, non-int level, and an inactive level on a dual
#     tensor are all rejected.
#
# No broadcast/backward dimensions apply: the op is unary over the dual tensor,
# performs no arithmetic, and its primal output is an aliasing view of the input
# (there is nothing to broadcast against or differentiate).

# Dual tensors can only be constructed over floating-point or complex primals
# (aten raises on int/bool primals). complex32 is excluded because the torch
# comparison helpers cannot materialize ComplexHalf on the reference device;
# complex64 covers the complex branch exactly.
UNPACK_DUAL_DTYPES = utils.ALL_FLOAT_DTYPES + [torch.complex64]

# Plain tensors have no such restriction, so every storage dtype is valid for
# the tangent-None path.
PLAIN_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES + [torch.complex64]
)

# Levels accepted by aten on plain tensors (no forward tangent) are 0, 1 and
# any higher value; 0 is always the documented level for dual tensors created
# inside the outermost ``dual_level()`` context.
UNPACK_DUAL_LEVELS = [0, 1, 3]

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_RANGE_SHAPES = [(), (256,), (7, 13, 29)]

_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_EMPTY_SHAPES = [(0,), (2, 0, 3)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._unpack_dual is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._unpack_dual callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_unpack_dual", getattr(flag_gems, "_unpack_dual", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_primal_view(res_primal, ref_primal, dual):
    # _unpack_dual returns the primal as an aliasing view (Tensor(a)): the
    # observable layout must match aten exactly and the result must share
    # storage with the candidate-side dual tensor.
    assert res_primal.dtype == ref_primal.dtype
    assert res_primal.shape == ref_primal.shape
    assert res_primal.stride() == ref_primal.stride()
    assert res_primal.storage_offset() == ref_primal.storage_offset()
    assert res_primal.data_ptr() == dual.data_ptr()


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_dual_tensor(shape, dtype):
    # The main forward-mode AD path: shape levels x every dual-capable dtype,
    # with values drawn from the default [-1, 1] range (negative and positive
    # for each dtype). Unpack at the level that created the dual and recover
    # both the primal view and the registered tangent.
    primal = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        # A dual tensor created with a tangent must yield a tensor tangent, not
        # None.
        assert isinstance(res_tangent_out, torch.Tensor)
        _assert_close(res_primal_out, ref_primal_out, dtype)
        _assert_close(res_tangent_out, ref_tangent_out, dtype)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_dual_tensor_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip bit-for-bit through both the primal view and the tangent.
    primal = tu.make_input(dtype, shape, value_range)
    tangent = tu.make_input(dtype, shape, value_range)
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        assert isinstance(res_tangent_out, torch.Tensor)
        tu.assert_result_close(res_primal_out, ref_primal_out)
        tu.assert_result_close(res_tangent_out, ref_tangent_out)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("level", UNPACK_DUAL_LEVELS)
@pytest.mark.parametrize("dtype", PLAIN_DTYPES)
def test__unpack_dual_plain_tensor(shape, level, dtype):
    # A plain tensor has no forward tangent at any level: aten still returns the
    # input as an aliasing view, and the tangent component must be None. This
    # exercises the tangent-None contract across every storage dtype.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_inp, level)
    res_primal_out, res_tangent_out = _resolve_gems_op()(inp, level)

    assert ref_tangent_out is None
    assert res_tangent_out is None
    _assert_close(res_primal_out, ref_primal_out, dtype)
    _assert_primal_view(res_primal_out, ref_primal_out, inp)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", PLAIN_DTYPES)
def test__unpack_dual_plain_tensor_value_ranges(shape, value_range, dtype):
    # The tangent-None path is also pure metadata: every spec value range
    # (including the exact int/bool comparisons) must round-trip through the
    # returned primal view.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_inp, 0)
    res_primal_out, res_tangent_out = _resolve_gems_op()(inp, 0)

    assert ref_tangent_out is None
    assert res_tangent_out is None
    tu.assert_result_close(res_primal_out, ref_primal_out)
    _assert_primal_view(res_primal_out, ref_primal_out, inp)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_non_contiguous(shape, dtype):
    # The aliasing primal view must preserve the exact strides and storage
    # offset of a non-contiguous primal. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout. The
    # tangent may be materialized with a different layout by the forward-AD
    # machinery, so only its values are compared.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    primal = base[..., ::2]
    ref_primal = ref_base[..., ::2]
    tangent = tu.make_input(dtype, primal.shape, ["-1", "1"])
    ref_tangent = utils.to_reference(tangent)
    assert not primal.is_contiguous()

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        assert res_primal_out.stride() == ref_primal_out.stride()
        assert res_primal_out.storage_offset() == ref_primal_out.storage_offset()
        assert res_primal_out.data_ptr() == primal.data_ptr()
        _assert_close(res_primal_out, ref_primal_out, dtype)
        _assert_close(res_tangent_out, ref_tangent_out, dtype)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_mutation(shape, dtype):
    # The returned primal is a true alias of the dual (Tensor(a)): writing
    # through it must be observable on the candidate-side primal, and the
    # reference must behave identically. The op itself never mutates anything.
    primal = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, _ = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, _ = _resolve_gems_op()(dual, level)

        ref_primal_out.fill_(2.5)
        res_primal_out.fill_(2.5)

        assert res_primal_out.data_ptr() == dual.data_ptr()
        _assert_close(res_primal_out, ref_primal_out, dtype)
        utils.gems_assert_equal(primal, ref_primal)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__unpack_dual_special_values(dtype):
    # A pure alias must preserve every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly through both the
    # primal and the tangent.
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_primal = utils.to_reference(values)
    tangent = torch.ones_like(values)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(values, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        utils.gems_assert_equal(res_primal_out, ref_primal_out, equal_nan=True)
        utils.gems_assert_equal(res_tangent_out, ref_tangent_out, equal_nan=True)
        assert (
            torch.signbit(res_primal_out[0]).item() == torch.signbit(values[0]).item()
        )
        assert (
            torch.signbit(res_primal_out[1]).item() == torch.signbit(values[1]).item()
        )


@pytest.mark._unpack_dual
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", UNPACK_DUAL_DTYPES)
def test__unpack_dual_empty(shape, dtype):
    # Empty tensors (0 elements) still carry a valid layout; the primal view
    # must preserve shape, strides and storage offset/data_ptr exactly and the
    # (also empty) tangent must round-trip.
    primal = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        ref_primal_out, ref_tangent_out = torch.ops.aten._unpack_dual(ref_dual, level)

        dual = torch.ops.aten._make_dual(primal, tangent, level)
        res_primal_out, res_tangent_out = _resolve_gems_op()(dual, level)

        _assert_close(res_primal_out, ref_primal_out, dtype)
        _assert_close(res_tangent_out, ref_tangent_out, dtype)
        _assert_primal_view(res_primal_out, ref_primal_out, dual)


@pytest.mark._unpack_dual
def test__unpack_dual_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python float hits the invalid
    # argument path and raises. The candidate must fail too rather than
    # silently accept scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._unpack_dual(3.14, 0)
    # The generated wrapper may fail on the first touch of the input (attribute
    # lookup, triton input validation or a dispatcher cast), so accept the
    # plausible Python failure modes; the point is that it must fail rather
    # than silently accept the scalar.
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(3.14, 0)


@pytest.mark._unpack_dual
def test__unpack_dual_rejects_non_int_level():
    # ``level`` is an int in the schema; a float is a cast error at the
    # dispatcher boundary and must be rejected by the candidate as well.
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._unpack_dual(ref_inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, 1.5)


@pytest.mark._unpack_dual
@pytest.mark.parametrize("bad_level", [-1, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__unpack_dual_rejects_inactive_level(dtype, bad_level):
    # A dual tensor carries its tangent only at the level that created it:
    # unpacking at any other level raises RuntimeError and the candidate must
    # reproduce the validation instead of silently returning a wrong tangent.
    # ``bad_level`` stays inactive by construction: -1 is never a valid level
    # and ``bad_level == 1`` differs from the context level, unless the context
    # happened to assign 1, in which case the next level is used instead.
    primal = tu.make_input(dtype, (4, 5), ["-1", "1"])
    tangent = tu.make_input(dtype, (4, 5), ["-1", "1"])
    ref_primal = utils.to_reference(primal)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_dual = torch.ops.aten._make_dual(ref_primal, ref_tangent, level)
        dual = torch.ops.aten._make_dual(primal, tangent, level)
        inactive = bad_level if bad_level != level else level + 1

        with pytest.raises(RuntimeError):
            torch.ops.aten._unpack_dual(ref_dual, inactive)
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(dual, inactive)
