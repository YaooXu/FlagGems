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
uses the same public callable with the actual ``out`` keyword.

dtype coverage
--------------
The positive cases use the shared regular floating-point dtypes, including
float64. The CUDA addmm path rejects integer, bool and FP8 operands; those
restrictions are covered by the negative dtype cases.

Value ranges
------------
The regular-operator spec's five ranges (``tu.selected_ranges()``) are swept.
Ranges keep their full magnitudes. The reference retains the input dtype at
each matrix product: a single fp64 upcast removes intermediate rounding and
overflow, changing the multi-product operator being tested.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------

# The CUDA addmm kernel accepts the regular floating-point dtypes.
_CHAIN_DTYPES = list(utils.ALL_FLOAT_DTYPES)

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
if tu.QUICK_MODE:
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

# Extreme workloads retain their declared bounds. Their oracle uses the original
# dtype because upcasting changes intermediate overflow and NaN propagation.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op(
        "chain_matmul", getattr(flag_gems, "chain_matmul", None)
    )


def _make_chain(shapes, dtype, value_range):
    return [tu.make_input(dtype, shape, value_range) for shape in shapes]


def _make_noncontig_chain(shapes, dtype, value_range):
    return [
        tu.make_input(dtype, (cols, rows), value_range).t() for rows, cols in shapes
    ]


def _to_ref(matrices):
    return [tu.to_reference(m) for m in matrices]


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
    tu.assert_result_close(res_out, ref_out.to(dtype))


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

    tu.assert_result_close(res_out, ref_out.to(dtype))


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
    res_ret = _resolve_gems_op()(inp, out=out)

    # The .out overload must write into and return the caller's buffer.
    assert ref_ret is ref_out
    assert res_ret is out
    tu.assert_result_close(out, ref_out.to(dtype))


# ---------------------------------------------------------------------------
# nan / inf
# ---------------------------------------------------------------------------


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", tu.selected_cases(_CHAIN_DTYPES))
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
@pytest.mark.parametrize("dtype", tu.selected_cases(_CHAIN_DTYPES))
def test_chain_matmul_backward(shapes, dtype):
    inp = [m.requires_grad_() for m in _make_chain(shapes, dtype, ["-1", "1"])]
    grad = tu.make_input(dtype, (shapes[0][0], shapes[-1][1]), ["-1", "1"])

    ref_inp = []
    for matrix in inp:
        ref_matrix = tu.to_reference(matrix.detach())
        ref_inp.append(ref_matrix.requires_grad_())
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    ref_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)
    for ref_g, shape in zip(ref_grads, shapes):
        assert ref_g.shape == shape

    # The candidate forward must match the native-dtype reference...
    res_out = _resolve_gems_op()(inp)
    tu.assert_result_close(res_out, ref_out.to(dtype))

    assert res_out.requires_grad
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
        _resolve_gems_op()(inp, out=res_bad)
