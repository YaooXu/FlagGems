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

# aten::adjoint(Tensor(a) self) -> Tensor(a) returns the conjugate-transpose
# (Hermitian adjoint) view of a matrix or batch of matrices: it is a zero-copy
# alias of the input equivalent to self.transpose(-2, -1).conj(). For real
# dtypes only the last two dimensions are swapped; for complex dtypes the view
# is additionally lazily conjugated. 0-D tensors degrade to a lazy conj() (with
# a deprecation warning) and 1-D tensors raise RuntimeError, so only shapes of
# rank >= 2 are used for the regular workload below.
ADJOINT_SHAPES = [
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

# adjoint is a dtype-agnostic zero-copy view (no arithmetic), so the result
# must match bit-for-bit. Complex dtypes exercise the lazy conjugation,
# float dtypes the transpose alone, and integer/bool dtypes confirm the op
# works for every storage dtype.
ADJOINT_DTYPES = (
    utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)


def _make_input(shape, dtype):
    if dtype.is_complex or dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.adjoint is registered; resolution order is: (1) override,
    # (2) the direct flag_gems.adjoint callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "adjoint", getattr(flag_gems, "adjoint", None)
    )


def _assert_view_semantics(res_out, ref_out):
    # adjoint returns an aliasing view (Tensor(a)); the observable layout and
    # conjugate flags must match aten exactly.
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_conj() == ref_out.is_conj()


def _assert_close(res_out, ref_out, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", ADJOINT_SHAPES)
@pytest.mark.parametrize("dtype", ADJOINT_DTYPES)
def test_adjoint(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_adjoint_non_contiguous(shape, dtype):
    # The transpose part of adjoint must preserve the strides of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    _assert_view_semantics(res_out, ref_out)


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", ADJOINT_DTYPES)
def test_adjoint_0d(dtype):
    # 0-D tensors cannot be transposed; aten degrades to a lazy conj() and
    # returns the conjugated scalar.
    inp = _make_input((), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_close(res_out, ref_out, dtype)
    assert res_out.shape == ref_out.shape


@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_adjoint_1d_raises(dtype):
    # 1-D tensors are neither matrices nor batches of matrices: aten raises
    # RuntimeError and the candidate must do the same.
    inp = _make_input((5,), dtype)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(ref_inp)
    gems_op = _resolve_gems_op()
    with pytest.raises(RuntimeError):
        gems_op(inp)
