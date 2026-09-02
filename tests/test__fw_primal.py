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

import os
import sys

# KernelGen's in-process verification (override_gems_op + pytest.main) runs
# pytest in-process with its own ``tests`` package earlier on sys.path than this
# checkout's ``tests`` package. With ``--import-mode=importlib`` pytest does not
# prepend the checkout root, so ``tests`` would resolve to the harness's package
# and ``from . import accuracy_utils`` would fail with ImportError during
# collection. Re-point the ``tests`` package at this file's directory before
# importing the helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests as _tests_pkg  # noqa: E402

if _HERE not in getattr(_tests_pkg, "__path__", []):
    sys.modules.pop("tests", None)
    import tests as _tests_pkg  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_fw_primal`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._fw_primal`` and ``-m _fw_primal`` both
# work.
setattr(
    pytest.mark,
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a) is the forward-mode
# AD view primitive: it returns an aliasing view of ``self`` that shares the
# input's storage (same shape, strides, storage offset, data_ptr and dtype)
# without any arithmetic. Level 0 is the documented, always-valid level; on
# plain tensors (no forward tangent registered) aten also accepts any higher
# level and still returns the input as a view, so the parametrization below
# covers 0, 1 and 3. Dual tensors created inside ``torch.autograd.forward_ad``
# are an internal AD-machinery edge case (level > 0 raises there) and are out of
# scope for a generated kernel op.
#
# The op is pure metadata manipulation, so every storage dtype is supported and
# the result compares bit-for-bit. Coverage follows the regular-operator spec
# adapted to a view op:
#   * shape levels: tu.selected_shapes() (ranks 0-8, selected by --quick);
#   * value ranges: tu.selected_ranges() over representative shapes, so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the view round-trips all of them);
#   * edge cases: non-contiguous (strided) inputs, empty tensors, nan/inf/±0.0
#     special values, and mutation through the returned alias;
#   * negative: non-tensor input and non-int level are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, performs no
# arithmetic, and its output is an aliasing view of the input (there is nothing
# to broadcast against or differentiate).

_FW_PRIMAL_LEVELS = [0, 1, 3]

_FW_PRIMAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
)

# Representative ranks for the full value-range sweep (0-dim, 1-dim, 3-dim);
# the shape-level sweep below already covers every rank in the active level.
_FW_PRIMAL_RANGE_SHAPES = [(), (256,), (7, 13, 29)]

_FW_PRIMAL_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_FW_PRIMAL_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_FW_PRIMAL_EMPTY_SHAPES = [(0,), (2, 0, 3)]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._fw_primal is registered; resolution order is: (1)
    # override, (2) the direct flag_gems._fw_primal callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_fw_primal", getattr(flag_gems, "_fw_primal", None)
    )


def _assert_view_semantics(res_out, ref_out, inp):
    # _fw_primal returns an aliasing view (Tensor(a)): the observable layout
    # must match aten exactly and the result must share storage with the input.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("level", _FW_PRIMAL_LEVELS)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal(shape, level, dtype):
    # Shape levels x level semantics x every storage dtype, with values drawn
    # from the default [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    tu.assert_result_close(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_value_ranges(shape, value_range, dtype):
    # A view never inspects or transforms the stored values, so the full spec
    # range sweep (negative, positive, extreme and degenerate ranges) must
    # round-trip bit-for-bit at the documented level 0.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    tu.assert_result_close(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_NONCONTIG_SHAPES)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_non_contiguous(shape, level, dtype):
    # The aliasing view must preserve the exact strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    tu.assert_result_close(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_MUTATION_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test__fw_primal_mutation(shape, dtype):
    # The result is a true alias of the input: writing through the returned
    # view must be observable on the candidate-side input tensor, and the
    # reference must behave identically. The reference runs on an independent
    # clone so the two aliases are validated separately.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())
    level = 0

    ref_out = torch.ops.aten._fw_primal(ref_inp, level)
    res_out = _resolve_gems_op()(inp, level)

    if dtype == torch.bool:
        res_out.fill_(True)
        ref_out.fill_(True)
    elif dtype.is_floating_point:
        res_out.fill_(2.5)
        ref_out.fill_(2.5)
    else:
        res_out.fill_(7)
        ref_out.fill_(7)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(inp, ref_inp)


@pytest.mark._fw_primal
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__fw_primal_special_values(dtype):
    # A pure view preserves every bit: signed zero, infinities and NaN
    # (including the NaN payload) must round-trip exactly.
    values = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1.5, -1.5, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values.clone())

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(values, 0)

    _assert_view_semantics(res_out, ref_out, values)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.signbit(res_out[0]).item() == torch.signbit(values[0]).item()
    assert torch.signbit(res_out[1]).item() == torch.signbit(values[1]).item()


@pytest.mark._fw_primal
@pytest.mark.parametrize("shape", _FW_PRIMAL_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _FW_PRIMAL_DTYPES)
def test__fw_primal_empty(shape, dtype):
    # Empty tensors (0 elements) still carry a valid layout; the view must
    # preserve shape, strides and storage offset/data_ptr exactly.
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._fw_primal(ref_inp, 0)
    res_out = _resolve_gems_op()(inp, 0)

    tu.assert_result_close(res_out, ref_out)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python float hits the invalid
    # argument path and raises. The candidate must fail too rather than
    # silently accept scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(3.14, 0)
    # The generated wrapper may fail on the first touch of the input (attribute
    # lookup, triton input validation or a dispatcher cast), so accept the
    # plausible Python failure modes; the point is that it must fail rather
    # than silently accept the scalar.
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(3.14, 0)


@pytest.mark._fw_primal
def test__fw_primal_rejects_non_int_level():
    # ``level`` is an int in the schema; a float is a cast error at the
    # dispatcher boundary and must be rejected by the candidate as well.
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten._fw_primal(ref_inp, 1.5)
    with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
        _resolve_gems_op()(inp, 1.5)
