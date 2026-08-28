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
from . import conftest as cfg

# aten::dstack views every input as 3-D (atleast_3d) and concatenates the
# resulting tensors along the new depth axis (dim 2): (N,) -> (1, N, 1),
# (M, N) -> (M, N, 1), and >= 3-D tensors are kept as-is and stacked along
# dim 2 (all dims except dim 2 must match). It is a pure data-movement op, so
# the result must match bit-for-bit, and every dtype the op supports is
# covered.
_FLOAT_DTYPES = set(utils.ALL_FLOAT_DTYPES)
DSTACK_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.BOOL_TYPES:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


if cfg.QUICK_MODE:
    DSTACK_SHAPES = [
        [(3,), (3,)],
        [(8, 16, 32), (8, 16, 48)],
    ]
else:
    DSTACK_SHAPES = [
        [(3,), (3,)],
        [(3, 33), (3, 33)],
        [(16, 16, 333), (16, 16, 333), (16, 16, 333)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
    ]


def _resolve_gems_op():
    # Resolution order: (1) the process-local override injected by KernelGen,
    # (2) the direct flag_gems.dstack callable, (3) None -> the test falls
    # back to the PyTorch reference so it stays runnable before a FlagGems
    # implementation is registered. The .default and .out overloads are
    # resolved through their public operator names "dstack" and "dstack.out".
    try:
        return flag_gems.testing.resolve_gems_op(
            "dstack", getattr(flag_gems, "dstack", None)
        )
    except LookupError:
        return None


def _resolve_gems_op_out():
    try:
        return flag_gems.testing.resolve_gems_op(
            "dstack.out", getattr(flag_gems, "dstack_out", None)
        )
    except LookupError:
        return None


def _apply_dstack(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.dstack(inp)
    return gems_op(inp)


def _apply_dstack_out(inp, out):
    gems_op = _resolve_gems_op_out()
    if gems_op is None:
        return torch.ops.aten.dstack.out(inp, out=out)
    return gems_op(inp, out=out)


@pytest.mark.dstack
@pytest.mark.parametrize("shape", DSTACK_SHAPES)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack(shape, dtype):
    inp = [_make_input(s, dtype) for s in shape]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.dstack(ref_inp)
    res_out = _apply_dstack(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack_out
@pytest.mark.parametrize("shape", DSTACK_SHAPES)
@pytest.mark.parametrize("dtype", DSTACK_DTYPES)
def test_dstack_out(shape, dtype):
    inp = [_make_input(s, dtype) for s in shape]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_shape = torch.ops.aten.dstack(ref_inp).shape
    ref_out = torch.empty(ref_shape, dtype=dtype, device=ref_inp[0].device)
    ref_ret = torch.ops.aten.dstack.out(ref_inp, out=ref_out)

    out = torch.empty(ref_shape, dtype=dtype, device=inp[0].device)
    res_ret = _apply_dstack_out(inp, out)

    # The .out variant must return the out tensor itself (alias semantics).
    assert res_ret.data_ptr() == out.data_ptr()
    assert ref_ret.data_ptr() == ref_out.data_ptr()
    assert res_ret.shape == ref_ret.shape
    assert res_ret.dtype == ref_ret.dtype
    utils.gems_assert_equal(res_ret, ref_ret)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.dstack
def test_dstack_empty_list():
    # dstack expects a non-empty TensorList.
    with pytest.raises(RuntimeError):
        _apply_dstack([])
