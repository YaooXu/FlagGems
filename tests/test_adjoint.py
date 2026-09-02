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

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package (and, for the sibling benchmark file, ``benchmark``) resolve
# to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# aten::adjoint(Tensor(a) self) -> Tensor(a) returns the conjugate-transpose
# (Hermitian adjoint) of a matrix or batch of matrices as a zero-copy aliasing
# view equivalent to self.transpose(-2, -1).conj(): for real dtypes only the
# last two dimensions are swapped, for complex dtypes the view is additionally
# lazily conjugated (the is_conj bit toggles). The view shares the input
# storage, so every storage dtype (float/complex/int/bool) is supported and the
# observed values round-trip exactly through the transposed/conjugated
# materialization. Degenerate inputs degrade differently: 0-D tensors fall back
# to a lazy conj() (deprecated in aten, the identity for real/int/bool dtypes)
# and 1-D tensors raise RuntimeError. The op is autograd-aware and an
# involution (adjoint is its own inverse), so d(adjoint(x))/dx == adjoint(dy).
#
# Coverage follows the regular-operator spec adapted to a view/metadata op:
#   * shape levels: rank >= 2 shapes selected by --quick plus representative
#     matrix / batch-of-matrices shapes (0-D/1-D get dedicated edge-case tests);
#   * value ranges: tu.selected_ranges() over representative ranks so every
#     supported dtype is exercised with negative, positive, extreme and
#     degenerate ranges (the aliasing view round-trips them exactly);
#   * edge cases: non-contiguous (strided) inputs, the conj-bit toggle, writing
#     through the returned alias, and nan/inf/+-0.0 special values;
#   * backward: autograd.grad() through the adjoint view against the analytic
#     gradient adjoint(dy) (broadcast does not apply to a unary view op);
#   * negative: 1-D inputs and non-tensor inputs raise on both the aten
#     reference and the candidate.
_ADJOINT_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.COMPLEX_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

# Representative matrix / batch-of-matrices shapes (2-D up to 5-D, small and
# mid-size) exercising contiguous storage.
_ADJOINT_SHAPES = [
    (2, 3),
    (32, 64),
    (256, 256),
    (2, 3, 4),
    (20, 320, 15),
    (4, 8, 16),
    (2, 3, 4, 5),
    (8, 16, 32, 64),
    (2, 3, 4, 5, 6),
]

# Representative ranks for the full value-range sweep (2-D, 3-D, 4-D).
_ADJOINT_RANGE_SHAPES = [(2, 3), (32, 64), (7, 13, 29)]
_ADJOINT_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_ADJOINT_TOGGLE_SHAPES = [(16, 32), (4, 8, 16)]
_ADJOINT_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]
_ADJOINT_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]


def _adjoint_test_shapes():
    # Shape levels (rank >= 2 only) merged with the representative matrix
    # shapes; 0-D/1-D are covered by the dedicated edge-case tests.
    return list(
        dict.fromkeys(
            _ADJOINT_SHAPES
            + [shape for shape in tu.selected_shapes() if len(shape) >= 2]
        )
    )


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.adjoint callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "adjoint", getattr(flag_gems, "adjoint", None)
    )


def _transposed_shape(shape):
    # Shape of adjoint(x): the last two dimensions are swapped.
    if len(shape) >= 2:
        return shape[:-2] + (shape[-1], shape[-2])
    return shape


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _assert_view_semantics(res_out, ref_out, inp):
    # adjoint returns an aliasing view (Tensor(a)): shape, strides, storage
    # offset, conjugation state and the shared storage must match aten exactly.
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _adjoint_test_shapes())
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint(shape, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_value_ranges(shape, value_range, dtype):
    # The op never transforms the stored values (beyond the lazy conjugate bit
    # toggle for complex dtypes), so the full spec range sweep must round-trip
    # exactly through the transposed/conjugated materialization.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_non_contiguous(shape, dtype):
    # The transpose part of adjoint must preserve the strides of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_TOGGLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_adjoint_toggle(shape, dtype):
    # The conj bit is a toggle: applying adjoint to an already-adjointed tensor
    # clears the bit and the materialized values come back to the base input
    # (adjoint is an involution).
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)

    inp = torch.ops.aten.adjoint(base)
    ref_inp = torch.ops.aten.adjoint(ref_base)
    assert inp.is_conj() == ref_inp.is_conj()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, base)
    assert not res_out.is_conj()
    assert not ref_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_adjoint_special_values(dtype):
    # adjoint is a pure view for real dtypes: +inf/-inf/nan/+-0.0 round-trip
    # unchanged through the transposed materialization; equal_nan=True in
    # assert_result_close tolerates the nan output.
    values = torch.tensor(
        [
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
            ]
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(values)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(values)

    _assert_view_semantics(res_out, ref_out, values)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adjoint_mutation(shape, dtype):
    # The result is a true alias of the input (Tensor(a)): writing through the
    # returned view stores into the shared storage and must be observable on
    # the candidate-side input. The reference runs on an independent clone so
    # the two aliases are validated separately.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    res_out = _resolve_gems_op()(inp)
    ref_out = torch.ops.aten.adjoint(ref_inp)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(inp, ref_inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_adjoint_backward(shape, dtype):
    # adjoint is an involution (its own inverse), so d(adjoint(x))/dx ==
    # adjoint(dy): the reference gradient must match the analytic value. The
    # candidate is validated on the same contract when it advertises autograd
    # support (a true view of a leaf carries requires_grad through the view
    # machinery; a materializing kernel would not).
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, _transposed_shape(shape), ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]
    expected_in_grad = torch.ops.aten.adjoint(ref_grad)
    tu.assert_result_close(ref_in_grad, expected_in_grad)

    # The candidate forward output must match the reference...
    res_out = _resolve_gems_op()(inp)
    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out, inp)

    # ...and, if the candidate advertises autograd support, its gradient must
    # match the analytic value too.
    if res_out.requires_grad:
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
        tu.assert_result_close(res_in_grad, expected_in_grad)


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _ADJOINT_DTYPES)
def test_adjoint_0d(dtype):
    # 0-D tensors cannot be transposed; aten degrades to a lazy conj() (with a
    # deprecation warning): the identity for real/int/bool dtypes and a lazy
    # conj view for complex dtypes. The candidate must match both the value and
    # the conjugation state.
    inp = tu.make_input(dtype, (), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.shape == ref_out.shape
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_adjoint_1d_raises(dtype):
    # 1-D tensors are neither matrices nor batches of matrices: aten raises
    # RuntimeError and the candidate must do the same.
    inp = tu.make_input(dtype, (5,), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(ref_inp)
    gems_op = _resolve_gems_op()
    with pytest.raises(RuntimeError):
        gems_op(inp)


@pytest.mark.adjoint
def test_adjoint_rejects_non_tensor():
    # The aten op requires a Tensor (a Python float hits a different overload
    # and raises); the candidate must fail too rather than silently accept
    # scalars.
    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
