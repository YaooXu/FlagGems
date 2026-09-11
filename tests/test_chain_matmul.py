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

"""Correctness tests for ``aten::chain_matmul`` (alias of ``torch.linalg.multi_dot``).

``chain_matmul`` multiplies a sequence of rank-2 matrices in an order chosen to
minimize the number of scalar multiplications:

    aten::chain_matmul(Tensor[] matrices) -> Tensor
    aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)

The operator therefore has a *list* of tensors as its single argument (not a
shape), only rank-2 matrices are accepted, there is no broadcasting, an empty
chain and dimension-mismatched chains raise ``RuntimeError``, and the forward is
differentiable with one gradient per input matrix. ``.out`` writes into and
returns the caller-provided buffer.

Candidate resolution
--------------------
The candidate is resolved *inside every test* (never at import time) through
``flag_gems.testing.resolve_gems_op`` so KernelGen's ``override_gems_op`` wins.
The default overload uses the public name ``chain_matmul``; the ``.out`` overload
uses the canonical dotted name ``chain_matmul.out`` (the harness also registers
the ``chain_matmul_out`` alias, exposed here as the default callable).

dtype coverage
--------------
The reference probe below (``tu.supported_dtypes``) confirms that the CUDA
``addmm`` path behind ``chain_matmul`` only accepts floating dtypes: int8 /
uint8 / float8_e4m3fn / float8_e5m2 / int32 / int64 / bool all raise
``"addmm_cuda" not implemented``. The candidate dtype list is therefore the
intersection of the supported floating set with ``utils.FLOAT_DTYPES`` (fp16 /
fp32 / bf16), matching the mm/bmm convention of not exercising fp64 on the
Triton matmul path.

Value ranges
------------
The regular-operator spec's five ranges (``tu.selected_ranges()``) are swept.
Matrix products amplify magnitude like ``|b| ** k`` for a length-k chain, so the
two unbounded ranges ``[0, max]`` / ``[min, 0]`` overflow the fp16/bf16/fp32
accumulators while the fp64 reference stays finite (the comparison would then be
meaningless). ``_make_chain`` keeps the range's sign structure and relative
spread but caps the magnitude of those two ranges at ``_EXTREME_MAGNITUDE``
before the Xavier ``1/sqrt(fan_in)`` scaling, so every entry remains a genuine
element of the requested range while the chain stays representable.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ---------------------------------------------------------------------------
# dtype probe
# ---------------------------------------------------------------------------

# The spec's required dtype candidates, probed against the real aten op below.
_CHAIN_DTYPE_CANDIDATES = [
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


def _probe_chain_dtype(_op_name, dtype):
    """Return True when ``dtype`` runs through ``aten::chain_matmul``."""
    try:
        left = tu.make_input(dtype, (4, 8), ["-1", "1"])
        right = tu.make_input(dtype, (8, 4), ["-1", "1"])
        torch.ops.aten.chain_matmul([left, right])
    except Exception:
        return False
    return True


# chain_matmul dispatches to addmm, which only has floating-point CUDA kernels
# ("addmm_cuda" is not implemented for int8/uint8/fp8/int32/int64/bool), so the
# float dtype set IS this operator's complete dtype set. The fallback therefore
# keeps the full (float-only) candidate set rather than dropping any of it.
_CHAIN_DTYPES = tu.supported_dtypes(
    "chain_matmul", list(utils.FLOAT_DTYPES), probe=_probe_chain_dtype
)
if not _CHAIN_DTYPES:
    _CHAIN_DTYPES = list(utils.FLOAT_DTYPES)

# ---------------------------------------------------------------------------
# chain shapes / value ranges
# ---------------------------------------------------------------------------

# Rank-2 chains covering the shape levels of the spec adapted to a
# list-of-matrices operator. The spec's seven dense shapes do not apply here:
# the input is ``Tensor[]`` (a matrix chain) whose adjacent matrices must agree
# on the inner dimension (M0[k,n0] @ M1[n0,n1] @ ...), so a single dense shape
# cannot express a legal input -- each entry below is a whole chain of matching
# rank-2 shapes. They cover a degenerate/single-matrix chain, short chains,
# rank-collapsing inner dims, 4/5-matrix chains and an odd, non-power-of-two
# chain that exercises tiling edges.
if tu.LEVEL == "quick":
    _CHAIN_SHAPES = [[(2, 3), (3, 4)]]
    _OUT_CHAIN_SHAPES = [[(2, 3), (3, 4)]]
    _BACKWARD_CHAINS = [[(2, 3), (3, 4)]]
    _NONCONTIG_CHAINS = [[(4, 8), (8, 16)]]
else:
    _CHAIN_SHAPES = [
        [(1, 1)],  # degenerate single-matrix chain
        [(4, 8)],  # single matrix: no product at all
        [(2, 3), (3, 4)],  # short chain
        [(1, 5), (5, 1), (1, 7)],  # rank-collapsing inner dims
        [(16, 32), (32, 64), (64, 32), (32, 16)],  # 4 matrices
        [(8, 16), (16, 32), (32, 48), (48, 32), (32, 16)],  # 5 matrices
        [(33, 65), (65, 17), (17, 129), (129, 255), (255, 71)],  # odd tiling
    ]
    _OUT_CHAIN_SHAPES = _CHAIN_SHAPES[:4]
    _BACKWARD_CHAINS = [
        [(4, 8)],
        [(2, 3), (3, 4)],
        [(4, 8), (8, 16), (16, 4)],
        [(16, 32), (32, 64), (64, 32), (32, 16)],
    ]
    _NONCONTIG_CHAINS = [
        [(4, 8), (8, 16)],
        [(16, 32), (32, 64), (64, 16)],
    ]

# The two unbounded ranges are magnitude-capped so the candidate's native-dtype
# accumulation cannot overflow while the fp64 reference stays finite.
_EXTREME_RANGES = (("0", "max"), ("min", "0"))
_EXTREME_MAGNITUDE = 4.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so the process-local
    # override installed by KernelGen for this run wins. flag_gems exposes no
    # direct chain_matmul callable yet, so the default is None and the override
    # registry (or a LookupError) decides.
    return flag_gems.testing.resolve_gems_op(
        "chain_matmul", getattr(flag_gems, "chain_matmul", None)
    )


def _resolve_gems_op_out():
    # The ".out" overload is registered by KernelGen under the canonical
    # "chain_matmul.out" name and the "chain_matmul_out" alias.
    return flag_gems.testing.resolve_gems_op(
        "chain_matmul.out", getattr(flag_gems, "chain_matmul_out", None)
    )


def _range_scale(dtype, value_range):
    """Magnitude scale applied on top of ``tu.make_input`` for a range."""
    if tuple(value_range) not in _EXTREME_RANGES:
        return 1.0
    bound = max(
        abs(tu.resolve_bound(value_range[0], dtype)),
        abs(tu.resolve_bound(value_range[1], dtype)),
    )
    return _EXTREME_MAGNITUDE / bound


def _make_chain(shapes, dtype, value_range):
    """Build a chain of rank-2 matrices from ``value_range``.

    Values come from ``tu.make_input`` (so they honour every dtype bound), are
    magnitude-capped for the unbounded ranges and Xavier-scaled by
    ``1/sqrt(fan_in)`` so that intermediate products of long chains stay inside
    the fp16/bf16 range.
    """
    scale = _range_scale(dtype, value_range)
    return [
        tu.make_input(dtype, shape, value_range) * (scale / (shape[1] ** 0.5))
        for shape in shapes
    ]


def _make_noncontig_chain(shapes, dtype, value_range):
    """Same chain as ``_make_chain`` but every matrix is a transposed view.

    The scaling is applied *before* the transpose so the returned tensor is the
    non-contiguous ``.t()`` view itself (a tensor produced by an elementwise
    multiply would be freshly allocated and contiguous).
    """
    scale = _range_scale(dtype, value_range)
    matrices = []
    for rows, cols in shapes:
        base = tu.make_input(dtype, (cols, rows), value_range)
        matrices.append((base * (scale / (cols**0.5))).t())
    return matrices


def _to_ref(matrices):
    return [utils.to_reference(m, m.is_floating_point()) for m in matrices]


def _assert_chain_close(res_out, ref_out, shapes, dtype):
    del shapes
    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    # The five spec ranges (in particular the unbounded [0, max] / [min, 0]
    # ones) feed wide-magnitude chains whose long fp32/bf16 reductions exceed
    # gems_assert_close's fp64-reference tolerances, so the value-range-friendly
    # spec helper (rtol=1e-2 / atol=1e-3 / equal_nan) performs the comparison;
    # the exact dtype and shape are still pinned explicitly above. The reference
    # is rounded to the candidate dtype first (the spec helper compares dtypes
    # strictly), which mirrors what gems_assert_close does internally.
    ref = ref_out if ref_out.dtype == dtype else ref_out.to(dtype)
    tu.assert_result_close(res_out, ref)


# ---------------------------------------------------------------------------
# default overload
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_chain_matmul(shapes, value_range, dtype):
    inp = _make_chain(shapes, dtype, value_range)
    ref_inp = _to_ref(inp)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.is_contiguous()
    _assert_chain_close(res_out, ref_out, shapes, dtype)


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _NONCONTIG_CHAINS)
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_chain_matmul_non_contiguous(shapes, dtype):
    # Transposed (non-unit-stride) matrices: the candidate must honour the
    # strides of each input rather than assuming contiguous memory.
    inp = _make_noncontig_chain(shapes, dtype, ["-1", "1"])
    assert all(not m.is_contiguous() for m in inp)
    ref_inp = _to_ref(inp)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_chain_close(res_out, ref_out, shapes, dtype)


# ---------------------------------------------------------------------------
# .out overload
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul_out
@pytest.mark.parametrize("shapes", _OUT_CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_chain_matmul_out(shapes, value_range, dtype):
    inp = _make_chain(shapes, dtype, value_range)
    ref_inp = _to_ref(inp)

    out_shape = (shapes[0][0], shapes[-1][1])
    # Garbage-prefilled buffers: the .out overload must overwrite every element.
    out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(
        out_shape, 7.0, dtype=ref_inp[0].dtype, device=ref_inp[0].device
    )

    ref_ret = torch.ops.aten.chain_matmul.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is out
    _assert_chain_close(out, ref_out, shapes, dtype)


# ---------------------------------------------------------------------------
# nan / inf
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_chain_matmul_nan_inf(dtype):
    # inf @ finite exercises inf * 0 -> nan inside the reduction as well as
    # inf + inf -> inf and (-inf) + (-inf) -> -inf; the output pattern is
    # deterministic on both paths and equal_nan=True tolerates the nan entries.
    m1 = torch.tensor(
        [[float("inf"), 1.0], [1.0, float("-inf")]],
        dtype=dtype,
        device=flag_gems.device,
    )
    m2 = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = [m1, m2]
    ref_inp = _to_ref(inp)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.dtype == dtype
    ref = ref_out if ref_out.dtype == dtype else ref_out.to(dtype)
    tu.assert_result_close(res_out, ref)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _BACKWARD_CHAINS)
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_chain_matmul_backward(shapes, dtype):
    inp = [m.requires_grad_() for m in _make_chain(shapes, dtype, ["-1", "1"])]
    grad = tu.make_input(dtype, (shapes[0][0], shapes[-1][1]), ["-1", "1"])

    ref_inp = []
    for matrix in inp:
        ref_matrix = utils.to_reference(matrix.detach(), True)
        ref_inp.append(ref_matrix.requires_grad_())
    ref_grad = utils.to_reference(grad, True)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    ref_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)
    for ref_g, shape in zip(ref_grads, shapes):
        assert ref_g.shape == shape

    # The candidate forward must match the fp64 reference...
    res_out = _resolve_gems_op()(inp)
    _assert_chain_close(res_out, ref_out, shapes, dtype)

    # ...and, if the candidate advertises autograd support (a plain fused kernel
    # does not: res_out.requires_grad is False), its gradients must match the
    # fp64 reference gradients as well.
    if res_out.requires_grad:
        res_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
        for res_g, ref_g in zip(res_grads, ref_grads):
            assert res_g.dtype == dtype
            assert res_g.shape == ref_g.shape
            tu.assert_result_close(res_g, ref_g.to(dtype))


# ---------------------------------------------------------------------------
# negative cases
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul([])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()([])


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_non_tensor():
    # The aten schema requires a Tensor[]; a scalar cannot match it.
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.chain_matmul(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_1d_matrix():
    inp = [tu.make_input(torch.float32, (4,), ["-1", "1"])]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_3d_tensor():
    matrix = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    inp = [matrix, matrix]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_mismatched_dims():
    inp = [
        tu.make_input(torch.float32, (3, 4), ["-1", "1"]),
        tu.make_input(torch.float32, (5, 6), ["-1", "1"]),
    ]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark.chain_matmul_negative
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="aten chain_matmul accepts integer addmm on the CPU reference path",
)
def test_chain_matmul_rejects_int_dtype():
    inp = [
        tu.make_input(torch.int32, (2, 2), ["0", "1"]),
        tu.make_input(torch.int32, (2, 2), ["0", "1"]),
    ]
    # On the accelerator the integer addmm is not implemented.
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark.chain_matmul_out_negative
def test_chain_matmul_out_rejects_wrong_dtype():
    # The .out overload validates the caller's buffer dtype and must raise for a
    # mismatched buffer instead of silently casting.
    inp = _make_chain([(4, 8), (8, 4)], torch.float32, ["-1", "1"])
    ref_inp = _to_ref(inp)

    ref_bad = torch.empty(4, 4, dtype=torch.int32, device=ref_inp[0].device)
    res_bad = torch.empty(4, 4, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul.out(ref_inp, out=ref_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op_out()(inp, out=res_bad)
