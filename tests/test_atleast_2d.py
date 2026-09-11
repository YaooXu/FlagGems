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

"""Correctness tests for ``aten::atleast_2d``.

``aten::atleast_2d`` is a pure view/identity operator: a 0-dim tensor becomes
``(1, 1)``, a 1-dim tensor ``(N,)`` becomes ``(1, N)`` (both views), and a
tensor with two or more dimensions is returned unchanged. No arithmetic is
performed, so the candidate must match the reference bit-for-bit, keep the
dtype, and alias the input storage. The test uses the regular-operator
value-range framework (``tests/test_utils.py``): the five shared value ranges
x the seven shape levels, plus broadcast-free metadata, nan/inf, complex,
backward and negative coverage.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ---------------------------------------------------------------------------
# Candidate resolution
# ---------------------------------------------------------------------------


def _resolve_candidate():
    """Resolve the candidate injected by KernelGen (or the FlagGems callable).

    Resolution is done *inside* the test function so an override installed by
    ``override_gems_op`` is honoured. ``flag_gems.atleast_2d`` is not a
    registered op yet, hence ``getattr(..., None)``: when neither an override
    nor a native callable exists we fall back to the PyTorch reference so the
    file stays runnable before an implementation lands.
    """
    try:
        return flag_gems.testing.resolve_gems_op(
            "atleast_2d", getattr(flag_gems, "atleast_2d", None)
        )
    except LookupError:
        return None


def _run(inp):
    """Apply the candidate (or the reference) to a Tensor or a list of Tensors."""
    candidate = _resolve_candidate()
    if candidate is None:
        if isinstance(inp, list):
            return torch.ops.aten.atleast_2d.Sequence(inp)
        return torch.ops.aten.atleast_2d(inp)
    return candidate(inp)


# ---------------------------------------------------------------------------
# Dtype coverage (probed with tu.supported_dtypes)
# ---------------------------------------------------------------------------

_EXTRA_CANDIDATES = [
    torch.float64,
    torch.int16,
    torch.bool,
    torch.complex64,
    torch.complex32,
]

# The spec requires int8/uint8/fp8 to be covered whenever the op supports them.
# atleast_2d is a pure view, so keep them in the candidate list explicitly
# (utils.ALL_INT_DTYPES only spans int16/int32/int64, and no shared set carries
# fp8). Same pattern as test_atleast_1d.py / test_atleast_3d.py.
_REQUIRED_EXTRA = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


def _dedup(dtypes):
    seen = set()
    out = []
    for dtype in dtypes:
        if dtype not in seen:
            seen.add(dtype)
            out.append(dtype)
    return out


_DTYPE_CANDIDATES = _dedup(
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + utils.COMPLEX_DTYPES
    + _EXTRA_CANDIDATES
    + _REQUIRED_EXTRA
)

# Fallback keeps the full candidate list (not just float32): if the probe
# cannot establish support it must not silently drop the required dtypes.
_SUPPORTED_DTYPES = (
    tu.supported_dtypes("atleast_2d", candidates=_DTYPE_CANDIDATES) or _DTYPE_CANDIDATES
)

_FP8_DTYPES = [
    d for d in (torch.float8_e4m3fn, torch.float8_e5m2) if d in _SUPPORTED_DTYPES
]
_VALUE_DTYPES = [d for d in _SUPPORTED_DTYPES if not d.is_complex]
_COMPLEX_DTYPES = [d for d in _SUPPORTED_DTYPES if d.is_complex]


def _assert_close(result, reference, dtype):
    """Compare with tu.assert_result_close, special-casing float8.

    This torch build cannot run ``assert_close`` on float8 CPU tensors, so the
    float8 path upcasts on the device first; the op is a pure view, so the
    upcast comparison is exact.
    """
    if dtype in _FP8_DTYPES:
        result = result.detach().to(torch.float32).cpu()
        reference = reference.detach().to(torch.float32).cpu()
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
    else:
        tu.assert_result_close(result, reference)


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


_VALUE_CASES = [(d, r) for d in _VALUE_DTYPES for r in tu.selected_ranges()]
_VALUE_CASE_IDS = [
    "{}-{}".format(str(d).replace("torch.", ""), "_".join(str(x) for x in r))
    for d, r in _VALUE_CASES
]

_SEQUENCE_DTYPES = {
    d
    for d in _VALUE_DTYPES
    if d in set(utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
}
_SEQUENCE_CASES = [(d, r) for d, r in _VALUE_CASES if d in _SEQUENCE_DTYPES]


# ---------------------------------------------------------------------------
# Value-range x shape grid (one parametrization combo == one Workload)
# ---------------------------------------------------------------------------


@pytest.mark.atleast_2d
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _VALUE_CASES, ids=_VALUE_CASE_IDS)
def test_atleast_2d_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = _run(inp)

    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    # atleast_2d returns a view: it must alias the input storage.
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.atleast_2d
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((), (1, 1)),
        ((1,), (1, 1)),
        ((5,), (1, 5)),
        ((3, 4), (3, 4)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 3, 4, 5), (2, 3, 4, 5)),
    ],
)
def test_atleast_2d_shape_metadata(shape, expected):
    # The dim boundary is what defines the op: <2 dims are promoted to 2 dims,
    # >= 2 dims are returned unchanged.
    inp = torch.arange(
        max(1, torch.Size(shape).numel()), dtype=torch.float32, device=flag_gems.device
    ).reshape(shape)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = _run(inp)

    assert tuple(ref_out.shape) == expected
    assert tuple(res_out.shape) == expected
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


# ---------------------------------------------------------------------------
# Sequence overload
# ---------------------------------------------------------------------------


@pytest.mark.atleast_2d_sequence
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    _SEQUENCE_CASES,
    ids=[
        "{}-{}".format(str(d).replace("torch.", ""), "_".join(str(x) for x in r))
        for d, r in _SEQUENCE_CASES
    ],
)
def test_atleast_2d_sequence(shape, dtype, value_range):
    # Mix a 0-dim scalar, a 1-dim tensor and the current shape so the sequence
    # overload exercises scalar -> (1, 1), 1-dim -> (1, N) and the >= 2-dim
    # identity path.
    inp = [
        tu.make_input(dtype, (), value_range),
        tu.make_input(dtype, (3,), value_range),
        tu.make_input(dtype, shape, value_range),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_2d.Sequence(ref_inp)
    res_out = _run(inp)

    assert isinstance(res_out, (list, tuple))
    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        assert res.data_ptr() == src.data_ptr()
        _assert_close(res, ref, dtype)


@pytest.mark.atleast_2d_sequence
def test_atleast_2d_sequence_empty():
    # A Tensor[] input may legitimately be empty: the reference returns an
    # empty list, and the candidate must return an empty list too.
    ref_out = torch.ops.aten.atleast_2d.Sequence([])
    res_out = _run([])
    assert len(ref_out) == 0
    assert len(res_out) == 0


# ---------------------------------------------------------------------------
# nan / inf preservation (float path)
# ---------------------------------------------------------------------------

_NAN_INF_VALUES = [
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


@pytest.mark.atleast_2d_nan_inf
@pytest.mark.parametrize("shape", [(), (9,), (3, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_2d_nan_inf(shape, dtype):
    # A pure view must preserve inf / -inf / nan and signed zeros unchanged
    # (tu.assert_result_close compares with equal_nan=True on the float path).
    values = _NAN_INF_VALUES[: 1 if shape == () else len(_NAN_INF_VALUES)]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device).reshape(shape)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = _run(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


# ---------------------------------------------------------------------------
# complex dtypes
# ---------------------------------------------------------------------------


@pytest.mark.atleast_2d_complex
@pytest.mark.parametrize("shape", [(), (5,), (2, 5), (2, 3, 4)])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_atleast_2d_complex(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    res_out = _run(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_close(res_out, ref_out)


# ---------------------------------------------------------------------------
# Backward (identity view: gradient of sum is all ones)
# ---------------------------------------------------------------------------

_BACKWARD_SHAPES = [(), (3,), (16, 64), (7, 13, 29)]


@pytest.mark.atleast_2d_backward
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_2d_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = utils.to_reference(inp)

    # atleast_2d is a view: d(sum(atleast_2d(x)))/dx is all ones in x's shape,
    # both on the shape-promoting (0-dim/1-dim) and identity paths.
    ref_out = torch.ops.aten.atleast_2d(ref_inp)
    ref_grad = torch.autograd.grad(ref_out.sum(), ref_inp)[0]
    tu.assert_result_close(ref_grad, torch.ones_like(ref_inp))

    res_out = _run(inp)
    tu.assert_result_close(res_out, ref_out)

    # A candidate that returns a plain (non-autograd-aware) tensor cannot be
    # differentiated; only check the gradient when the graph exists.
    if res_out.requires_grad:
        res_grad = torch.autograd.grad(res_out.sum(), inp)[0]
        tu.assert_result_close(res_grad, torch.ones_like(inp))


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.atleast_2d_negative
def test_atleast_2d_rejects_non_tensor():
    # The aten op requires a Tensor / Tensor[]; scalars hit the argument type
    # check and raise.
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_2d(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten.atleast_2d.Sequence(
            [torch.zeros(2, device=flag_gems.device), 3.14]
        )

    candidate = _resolve_candidate()
    if candidate is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            candidate(3.14)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            candidate([torch.zeros(2, device=flag_gems.device), 3.14])
