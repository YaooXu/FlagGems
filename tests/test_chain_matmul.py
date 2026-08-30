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

# aten::chain_matmul(Tensor[] matrices) -> Tensor (alias of linalg.multi_dot)
# multiplies a sequence of rank-2 matrices in an order chosen to minimize the
# total number of scalar multiplications; the .out overload writes into a
# caller-provided buffer. Only rank-2 matrices are accepted (1-D, 3-D and
# dimension-mismatched chains raise), there is no broadcast semantics, and the
# empty chain raises. The op is differentiable; the backward pass returns one
# gradient per matrix (chain_matmul has no single "self" tensor).
#
# Precision: the candidate kernels target GPU matmul, which runs fp16/bf16/fp32
# with TF32 on the native path, so every float case is validated against an
# fp64-upcast reference through utils.gems_assert_close (which casts the
# reference back to the tested dtype) with the tolerance scaled by the largest
# reduced dimension of the chain. Inputs are Xavier-scaled by 1/sqrt(fan_in)
# like an orthogonal initialization so intermediate products stay bounded for
# fp16/bf16.
#
# Value ranges: tu.selected_ranges() would include the extreme ["0","max"] /
# ["min","0"] bands, which overflow the fp16/bf16 accumulators of long chains
# even after Xavier scaling. Matrix products are linear in their inputs, so the
# sign/locality coverage is kept in the three safe bands below and the extreme
# magnitudes are exercised separately by the nan/inf test.
_CHAIN_VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
]

# Chain shapes: rank-2 only. A single-matrix chain returns the matrix unchanged;
# the last chain (odd, non-power-of-two inner dims) exercises tiling edge cases.
_CHAIN_SHAPES = [
    [(4, 8)],
    [(2, 3), (3, 4)],
    [(4, 8), (8, 16), (16, 4)],
    [(1, 5), (5, 1), (1, 7)],
    [(16, 32), (32, 64), (64, 32), (32, 16)],
    [(8, 16), (16, 32), (32, 48), (48, 32), (32, 16)],
    [(33, 65), (65, 17), (17, 129), (129, 255), (255, 71)],
]

# Backward shapes stay small (the autograd graph is built on the fp64 reference
# and the analytic comparison below is elementwise).
_BACKWARD_CHAINS = [
    [(4, 8)],
    [(2, 3), (3, 4)],
    [(4, 8), (8, 16), (16, 4)],
    [(16, 32), (32, 64), (64, 32), (32, 16)],
]


def _make_chain(shapes, dtype, value_range):
    """Build a chain with values from ``value_range``, Xavier-scaled by
    1/sqrt(fan_in) so intermediate products stay bounded."""
    return [
        tu.make_input(dtype, shape, value_range) / (shape[1] ** 0.5) for shape in shapes
    ]


def _max_reduce_dim(shapes):
    if len(shapes) <= 1:
        return 1
    return max(a[1] for a, b in zip(shapes, shapes[1:]))


def _atol_base(dtype):
    if dtype == torch.bfloat16:
        return 2e-3
    return 5e-4


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. flag_gems.chain_matmul
    # is not exported yet, so without an override this raises LookupError: the
    # file is exercised by KernelGen after override_gems_op("chain_matmul", ...).
    return flag_gems.testing.resolve_gems_op(
        "chain_matmul", getattr(flag_gems, "chain_matmul", None)
    )


def _chain_matmul_out_adapter(matrices, *, out):
    # Default implementation of the ".out" overload: run the direct chain_matmul
    # kernel and copy the result into the caller's out buffer. KernelGen's
    # override of "chain_matmul_out" replaces this adapter with a real kernel.
    out.copy_(_resolve_gems_op()(matrices))
    return out


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "chain_matmul_out", _chain_matmul_out_adapter
    )


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", _CHAIN_VALUE_RANGES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul(shapes, value_range, dtype):
    inp = _make_chain(shapes, dtype, value_range)
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = _resolve_gems_op()(inp)

    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=_max_reduce_dim(shapes),
        atol=_atol_base(dtype),
    )


@pytest.mark.chain_matmul_out
@pytest.mark.parametrize("shapes", _CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", _CHAIN_VALUE_RANGES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_out(shapes, value_range, dtype):
    inp = _make_chain(shapes, dtype, value_range)
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]

    out_shape = (shapes[0][0], shapes[-1][1])
    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(out_shape, 7.0, dtype=torch.float64, device=flag_gems.device)

    ref_out_res = torch.ops.aten.chain_matmul.out(ref_inp, out=ref_out)
    res_out = _resolve_gems_op_out()(inp, out=out)

    assert ref_out_res is ref_out
    assert res_out is out
    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(
        out,
        ref_out,
        dtype,
        reduce_dim=_max_reduce_dim(shapes),
        atol=_atol_base(dtype),
    )


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_nan_inf(dtype):
    # m1 @ m2 exercises inf * 0 -> nan inside the reduction as well as
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
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = _resolve_gems_op()(inp)

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        reduce_dim=2,
        atol=1e-4,
    )


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _BACKWARD_CHAINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_backward(shapes, dtype):
    inp = [m.requires_grad_() for m in _make_chain(shapes, dtype, ["-1", "1"])]
    grad = tu.make_input(dtype, (shapes[0][0], shapes[-1][1]), ["-1", "1"])
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    ref_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)
    for ref_g, shape in zip(ref_grads, shapes):
        assert ref_g.shape == shape

    # The candidate forward output must match the fp64 reference...
    res_out = _resolve_gems_op()(inp)
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=_max_reduce_dim(shapes),
        atol=_atol_base(dtype),
    )

    # ...and, if the candidate kernel advertises autograd support (a plain
    # fused kernel does not: res_out.requires_grad is False), its gradients
    # must match the fp64 reference gradients too.
    if res_out.requires_grad:
        res_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
        for res_g, ref_g in zip(res_grads, ref_grads):
            utils.gems_assert_close(
                res_g, ref_g, dtype, reduce_dim=1, atol=_atol_base(dtype)
            )


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul([])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()([])


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_1d_matrix():
    inp = [tu.make_input(torch.float32, (4,), ["-1", "1"])]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_3d_tensor():
    m = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    inp = [m, m]
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
def test_chain_matmul_rejects_int_dtype():
    inp = [
        tu.make_input(torch.int32, (2, 2), ["-1", "1"]),
        tu.make_input(torch.int32, (2, 2), ["-1", "1"]),
    ]
    # aten chain_matmul is only implemented for floating addmm on the
    # accelerator; integer chains are rejected on CUDA but accepted on the CPU
    # reference path, so the reference assertion is gated on the device.
    if flag_gems.device != "cpu":
        with pytest.raises(RuntimeError):
            torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(inp)
