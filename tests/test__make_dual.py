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
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by --quick);
#   * value ranges: tu.selected_ranges() over representative shapes, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the aliasing view round-trips all of them bit-for-bit,
#     and the tangent is preserved unchanged);
#   * edge cases: non-contiguous (strided) primals, mutation through the
#     returned alias, and nan/inf/±0.0 special values;
#   * negative: int/bool primal, tangent/primal size mismatch, and a level that
#     is not active all raise.
#
# No broadcast/backward dimensions apply: the tangent must match the primal's
# size exactly (aten rejects any broadcast shape), and _make_dual is a
# forward-AD construction primitive with no backward defined.

# Primal must be floating-point or complex. complex32 is excluded because the
# torch comparison helpers cannot materialize ComplexHalf on the reference
# device; complex64 covers the complex branch exactly.
MAKE_DUAL_DTYPES = utils.ALL_FLOAT_DTYPES + [torch.complex64]

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_MAKE_DUAL_RANGE_SHAPES = [(), (256,), (7, 13, 29)]

_MAKE_DUAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_MAKE_DUAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]


# Resolved inside each test (never at import time) so that the process-local
# override installed by KernelGen for this run wins. The default stays None
# until flag_gems._make_dual is registered; resolution order is: (1)
# override, (2) the direct flag_gems._make_dual callable, (3) LookupError.
def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "_make_dual", getattr(flag_gems, "_make_dual", None)
    )


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


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
    _assert_close(res_primal, ref_primal, dtype)
    _assert_close(res_tangent, ref_tangent_out, dtype)
    _assert_close(res_tangent, ref_tangent, dtype)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", MAKE_DUAL_DTYPES)
def test__make_dual(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
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
@pytest.mark.parametrize("dtype", MAKE_DUAL_DTYPES)
def test__make_dual_value_ranges(shape, value_range, dtype):
    # The op never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip bit-for-bit and the tangent must survive unchanged.
    inp = tu.make_input(dtype, shape, value_range)
    tangent = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)
    ref_tangent = utils.to_reference(tangent)

    with dual_level() as level:
        ref_out = torch.ops.aten._make_dual(ref_inp, ref_tangent, level)
        res_out = _resolve_gems_op()(inp, tangent, level)

        _assert_view_semantics(res_out, ref_out, inp)
        res_primal, res_tangent = torch.autograd.forward_ad.unpack_dual(res_out)
        ref_primal, ref_tangent_out = torch.autograd.forward_ad.unpack_dual(ref_out)
        tu.assert_result_close(res_primal, ref_primal)
        tu.assert_result_close(res_tangent, ref_tangent_out)
        tu.assert_result_close(res_tangent, ref_tangent)


@pytest.mark._make_dual
@pytest.mark.parametrize("shape", _MAKE_DUAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", MAKE_DUAL_DTYPES)
def test__make_dual_non_contiguous(shape, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous primal. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    tangent = tu.make_input(dtype, inp.shape, ["-1", "1"])
    ref_tangent = utils.to_reference(tangent)
    assert not inp.is_contiguous()

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
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    tangent = tu.make_input(dtype, shape, ["-1", "1"])
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
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
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

        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
        assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
        assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._make_dual
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test__make_dual_rejects_non_float_primal(dtype):
    # Forward-mode dual tensors only support floating-point/complex storage:
    # aten raises on int/bool primals (the internal assert requires both primal
    # and tangent to be floating point or complex) and the candidate must too.
    with dual_level() as level:
        inp = tu.make_input(dtype, (4, 5), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
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
def test__make_dual_rejects_tangent_size_mismatch():
    # The tangent must have exactly the primal's size: aten rejects any other
    # shape (broadcasting is not defined for forward tangents) and the
    # candidate must reproduce the validation.
    with dual_level() as level:
        inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
        tangent = tu.make_input(torch.float32, (3, 7), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten._make_dual(
                utils.to_reference(inp), utils.to_reference(tangent), level
            )
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            _resolve_gems_op()(inp, tangent, level)


@pytest.mark._make_dual
def test__make_dual_rejects_inactive_level():
    # The named level must be live: outside dual_level() aten rejects any level
    # index with RuntimeError and the candidate must reproduce the validation
    # instead of silently ignoring the level.
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    tangent = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_dual(
            utils.to_reference(inp), utils.to_reference(tangent), 0
        )
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, tangent, 0)
