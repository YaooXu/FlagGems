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

# aten::atleast_3d is a pure view/identity op: 0-dim tensors become (1, 1, 1),
# 1-dim tensors (N,) become (1, N, 1), 2-dim tensors (M, N) become (M, N, 1)
# and tensors with three or more dimensions are returned unchanged. No
# arithmetic is performed, so the result must match bit-for-bit and alias the
# input. It has two overloads:
#   * atleast_3d(Tensor) -> Tensor
#   * atleast_3d.Sequence(Tensor[]) -> Tensor[]
# both of which are exercised below.
#
# Coverage follows the regular-operator test spec:
#   * shape levels: the shared tu.selected_shapes() levels (the spec's seven
#     shapes, one per rank 0..5) in the base dtype test, plus a representative
#     rank-complete set for the value-range / sequence sweeps so the grid stays
#     fast;
#   * value ranges: the full tu.selected_ranges() sweep ([-1,1], [0,1], [-1,0],
#     [0,max], [min,0]) per supported dtype (values must round-trip exactly
#     through the view - the earlier randn-only generation is migrated onto this
#     framework). Ranges that an unsigned dtype cannot represent are dropped;
#   * dtypes: the spec's required list first (int8 / uint8 / fp8 are hard
#     requirements and are probed on the active backend before being used),
#     then the shared float/int/bool sets, plus a dedicated complex case;
#   * no broadcast dimension exists (the op is unary), so broadcast is skipped;
#   * edge cases: nan/inf/+-0.0 passthrough, complex tensors, an empty
#     TensorList, and alias/view semantics (the result shares the input's
#     storage);
#   * backward: autograd.grad() of sum(atleast_3d(x)) is all-ones in x's shape;
#   * negative: a non-tensor argument (including a non-tensor list element)
#     must raise on both the reference and the candidate path.
#
# Both overloads are resolved through the shared public operator name
# "atleast_3d" via flag_gems.testing.resolve_gems_op(...) inside each test
# (never at import time), so the process-local override injected by KernelGen
# wins. When neither an override nor a native implementation exists yet the
# tests fall back to the PyTorch reference so the file stays runnable.

_FP8_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

# The spec's required dtype list first (int8 / uint8 / fp8 are hard
# requirements when the backend supports them), then the shared float/int/bool
# sets. The probe below removes anything the active backend cannot store.
_DTYPE_CANDIDATES = []
for _dtype in (
    [torch.int8, torch.uint8]
    + sorted(_FP8_DTYPES, key=str)
    + list(utils.ALL_FLOAT_DTYPES)
    + list(utils.ALL_INT_DTYPES)
    + list(utils.BOOL_TYPES)
):
    if _dtype not in _DTYPE_CANDIDATES:
        _DTYPE_CANDIDATES.append(_dtype)


def _probe_dtypes():
    """Keep only the dtypes the active backend can actually pass through.

    ``tu.supported_dtypes`` builds a single tiny input per dtype and treats any
    exception as "unsupported"; atleast_3d is a view, so a storage dtype that
    cannot even be allocated is removed here. Falls back to the shared
    float/int/bool sets so the file never collects zero cases.
    """
    supported = []
    for dtype in _DTYPE_CANDIDATES:
        try:
            x = tu.make_input(dtype, (4,), ["0", "1"])
            out = torch.ops.aten.atleast_3d(x)
        except Exception:
            continue
        if out.dtype == dtype:
            supported.append(dtype)
    if not supported:
        supported = (
            list(utils.ALL_FLOAT_DTYPES)
            + list(utils.ALL_INT_DTYPES)
            + list(utils.BOOL_TYPES)
        )
    return supported


ATLEAST_3D_DTYPES = _probe_dtypes()

# Shape levels: the shared spec set (seven ranks, quick keeps a single 3-D
# shape). These are used for the base dtype test.
_SHAPES = tu.selected_shapes()

# Representative rank-complete shapes for the range/sequence sweeps: one shape
# per rank 0..4 plus a rank-4 and rank-5 case. Small, so the 5-range x dtype
# grid stays fast while still covering every rank the op accepts.
_RANGE_SHAPES = [
    (),
    (1,),
    (256,),
    (4, 5),
    (8, 16, 32),
    (4, 5, 6, 7),
    (2, 3, 4, 5, 6),
]

# The sequence overload mixes a 0-dim scalar, a 1-dim tensor, a 2-dim tensor
# and the current shape so that all four view paths (scalar -> (1,1,1),
# 1-dim -> (1,N,1), 2-dim -> (M,N,1), >= 3-dim identity) are exercised.
_SEQUENCE_SHAPES = list(_RANGE_SHAPES)

# Backward shapes stay small; the autograd graph is compared element-wise.
_BACKWARD_SHAPES = [(), (3,), (4, 5), (16, 64), (7, 13, 29)]


def _valid_ranges(dtype):
    """The value ranges from the shared spec that ``dtype`` can represent.

    Unsigned integers cannot represent the negative-only ranges, so those
    combinations are dropped instead of being fed to a generator that would
    reject them.
    """
    pairs = []
    for value_range in tu.selected_ranges():
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        if dtype == torch.uint8 and (low < 0 or high < 0):
            continue
        pairs.append((dtype, value_range))
    return pairs


_DTYPE_RANGE_PAIRS = []
for _dtype in ATLEAST_3D_DTYPES:
    _DTYPE_RANGE_PAIRS.extend(_valid_ranges(_dtype))


def _resolve_named_gems_op(name):
    """Resolve one operator/overload name through resolve_gems_op.

    Resolution order: (1) the process-local override installed by KernelGen,
    (2) the direct flag_gems callable for that name, (3) None -> the caller
    falls back to the PyTorch reference.
    """
    default = getattr(flag_gems, name.replace(".", "_"), None)
    if default is None:
        default = getattr(flag_gems, name, None)
    try:
        return flag_gems.testing.resolve_gems_op(name, default)
    except LookupError:
        return None


def _resolve_gems_op():
    return _resolve_named_gems_op("atleast_3d")


def _resolve_gems_op_sequence():
    # The verify harness may register the overload as "atleast_3d.Sequence",
    # "atleast_3d_sequence" or a single "atleast_3d" callable handling both.
    for name in ("atleast_3d.Sequence", "atleast_3d_sequence", "atleast_3d"):
        op = _resolve_named_gems_op(name)
        if op is not None:
            return op
    return None


def _apply_atleast_3d(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone. The aten
        # packet auto-selects the Sequence overload when handed a list.
        return torch.ops.aten.atleast_3d(inp)
    return gems_op(inp)


def _apply_atleast_3d_sequence(inp):
    gems_op = _resolve_gems_op_sequence()
    if gems_op is None:
        return torch.ops.aten.atleast_3d.Sequence(inp)
    return gems_op(inp)


def _assert_result(res_out, ref_out, dtype):
    """Compare shape/dtype/values; fp8 uses the device-resident exact helper
    (torch.testing has no CPU fp8 comparison support), everything else goes
    through the tolerance-aware value-range helper (exact for int/bool)."""
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if dtype in _FP8_DTYPES:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        tu.assert_result_close(res_out, ref_out)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d(shape, dtype):
    # Shape levels x every supported dtype (incl. the required int8/uint8/fp8)
    # with values drawn from a non-degenerate [-1, 1] range.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    _assert_result(res_out, ref_out, dtype)
    # A view/identity op must alias its input (Tensor(a)).
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", _RANGE_SHAPES)
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test_atleast_3d_value_ranges(shape, dtype, value_range):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate constant ranges) must round-trip
    # exactly through the shape-changing view.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    _assert_result(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("shape", _SEQUENCE_SHAPES)
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_sequence(shape, dtype):
    # The Tensor[] overload must apply the same view per element: scalar ->
    # (1,1,1), 1-dim -> (1,N,1), 2-dim -> (M,N,1) and >= 3-dim identity.
    inp = [
        tu.make_input(dtype, (), ["-1", "1"]),
        tu.make_input(dtype, (3,), ["-1", "1"]),
        tu.make_input(dtype, (4, 5), ["-1", "1"]),
        tu.make_input(dtype, shape, ["-1", "1"]),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = _apply_atleast_3d_sequence(inp)

    assert len(res_out) == len(ref_out) == 4
    for res, ref, src in zip(res_out, ref_out, inp):
        _assert_result(res, ref, dtype)
        # Each result is a view of its own input.
        assert res.data_ptr() == src.data_ptr()


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test_atleast_3d_sequence_value_ranges(dtype, value_range):
    # Range sweep for the Tensor[] overload over the three shape-changing
    # paths (0-dim / 1-dim / 2-dim).
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, (3,), value_range),
        tu.make_input(dtype, (4, 5), value_range),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = _apply_atleast_3d_sequence(inp)

    assert len(res_out) == len(ref_out) == 3
    for res, ref, src in zip(res_out, ref_out, inp):
        _assert_result(res, ref, dtype)
        assert res.data_ptr() == src.data_ptr()


@pytest.mark.atleast_3d_sequence
def test_atleast_3d_sequence_empty():
    # An empty Tensor[] is legitimate: the reference returns an empty list and
    # the candidate must do the same (atleast_3d.Sequence([]) does not raise).
    ref_out = torch.ops.aten.atleast_3d.Sequence([])
    res_out = _apply_atleast_3d_sequence([])
    assert len(ref_out) == 0
    assert len(res_out) == 0


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_atleast_3d_nan_inf(dtype):
    # Values pass through a view untouched: nan/inf/-inf and signed zeros must
    # be preserved (the float comparison path uses equal_nan=True). 1e30
    # overflows to inf in fp16/bf16 identically on both paths.
    inp = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    _assert_result(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_atleast_3d_complex(dtype):
    # atleast_3d also supports complex tensors (a pure view: real and imaginary
    # parts pass through untouched). One negative-and-positive range per dtype
    # suffices because no arithmetic is performed.
    inp = tu.make_input(dtype, (2, 5), ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    _assert_result(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d_backward
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_3d_backward(shape, dtype):
    # atleast_3d is a view: grad(sum(atleast_3d(x))) is all-ones in x's shape
    # on both the shape-changing (0-dim/1-dim/2-dim) and the identity paths.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = inp.detach().clone().requires_grad_()

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    ref_grad = torch.autograd.grad(ref_out.sum(), ref_inp)[0]
    tu.assert_result_close(ref_grad, torch.ones_like(ref_inp))

    # The candidate forward must match the reference...
    res_out = _apply_atleast_3d(inp)
    _assert_result(res_out, ref_out, dtype)

    # ...and, when the candidate view is autograd-aware (a compiled kernel that
    # returns a plain tensor is not), its gradient must match too.
    if res_out.requires_grad:
        res_grad = torch.autograd.grad(res_out.sum(), inp)[0]
        assert res_grad.shape == inp.shape
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.atleast_3d_negative
def test_atleast_3d_rejects_non_tensor():
    # The aten op requires a Tensor (the Tensor overload) / a TensorList whose
    # elements are Tensors (the Sequence overload); a Python float must raise on
    # both the reference and the candidate path rather than being accepted.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_3d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )

    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_op(3.14)

    gems_seq_op = _resolve_gems_op_sequence()
    if gems_seq_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            gems_seq_op([torch.zeros(2, device=flag_gems.device), 3.14])
