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

# aten::atleast_3d is a pure view/identity op: 0-dim tensors are reshaped to
# (1, 1, 1), 1-dim tensors to (1, N, 1), 2-dim tensors to (M, N, 1) (all
# views), and tensors with three or more dimensions are returned as-is. No
# arithmetic is performed, so the result must match bit-for-bit, alias the
# input, and every dtype the op supports is covered.
_FLOAT_DTYPES = set(utils.ALL_FLOAT_DTYPES)
ATLEAST_3D_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.BOOL_TYPES:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolution order: (1) the process-local override injected by KernelGen,
    # (2) the direct flag_gems.atleast_3d callable, (3) None -> the test falls
    # back to the PyTorch reference so it stays runnable before a FlagGems
    # implementation is registered. Both the .default and .Sequence overloads
    # are resolved through the shared public operator name "atleast_3d".
    try:
        return flag_gems.testing.resolve_gems_op(
            "atleast_3d", getattr(flag_gems, "atleast_3d", None)
        )
    except LookupError:
        return None


def _apply_atleast_3d(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.atleast_3d(inp)
    return gems_op(inp)


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.atleast_3d(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # atleast_3d is a view op: the result must alias the input.
    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.atleast_3d_sequence
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ATLEAST_3D_DTYPES)
def test_atleast_3d_sequence(shape, dtype):
    # Mix a 0-dim scalar, a 1-dim tensor, a 2-dim tensor and the current shape
    # so the sequence overload exercises all four paths: scalar -> (1, 1, 1),
    # 1-dim -> (1, N, 1), 2-dim -> (M, N, 1) and the >= 3-dim identity.
    inp = [
        _make_input((), dtype),
        _make_input((3,), dtype),
        _make_input((4, 5), dtype),
        _make_input(shape, dtype),
    ]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.atleast_3d.Sequence(ref_inp)
    res_out = _apply_atleast_3d(inp)

    assert len(res_out) == len(ref_out)
    for res, ref, src in zip(res_out, ref_out, inp):
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        # atleast_3d is a view op: each result must alias its input.
        assert res.data_ptr() == src.data_ptr()
        utils.gems_assert_equal(res, ref)
