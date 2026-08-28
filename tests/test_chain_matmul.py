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

# chain_matmul (alias of torch.linalg.multi_dot) multiplies a sequence of 2-D
# matrices in an order chosen to minimize the total number of scalar
# multiplications. Only matrices (rank-2) are supported. Each element below is
# a chain: a list of (rows, cols) shapes that multiply in sequence.
_CHAIN_SHAPES = [
    [(4, 8)],
    [(2, 3), (3, 4)],
    [(4, 8), (8, 16), (16, 4)],
    [(1, 5), (5, 1), (1, 7)],
    [(16, 32), (32, 64), (64, 32), (32, 16)],
    [(8, 16), (16, 32), (32, 48), (48, 32), (32, 16)],
    [(33, 65), (65, 17), (17, 129), (129, 255), (255, 71)],
]


# The chain multiplication order groups operands by magnitude so intermediate
# products can be large. Native GPU matmul runs with TF32 enabled, so to keep
# the comparison against a float64 reference deterministic we scale every
# matrix like an orthogonal (Xavier) initialization: values ~ N(0, 1 / fan_in).
def _make_chain(shapes, dtype):
    return [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) / (shape[1] ** 0.5)
        for shape in shapes
    ]


def _max_reduce_dim(shapes):
    if len(shapes) <= 1:
        return 1
    return max(a[1] for a, b in zip(shapes, shapes[1:]))


def _atol_base(dtype):
    if dtype == torch.bfloat16:
        return 2e-3
    return 5e-4


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _CHAIN_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul(shapes, dtype):
    inp = _make_chain(shapes, dtype)
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    gems_op = flag_gems.testing.resolve_gems_op(
        "chain_matmul", getattr(flag_gems, "chain_matmul", None)
    )
    res_out = gems_op(inp)

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
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_out(shapes, dtype):
    inp = _make_chain(shapes, dtype)
    ref_inp = [utils.to_reference(m, upcast=True) for m in inp]

    out_shape = (shapes[0][0], shapes[-1][1])
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.empty(out_shape, dtype=torch.float64, device=flag_gems.device)

    ref_out_res = torch.ops.aten.chain_matmul.out(ref_inp, out=ref_out)
    gems_op = flag_gems.testing.resolve_gems_op(
        "chain_matmul_out", getattr(flag_gems, "chain_matmul", None)
    )
    res_out = gems_op(inp, out=out)

    assert res_out is out
    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    assert ref_out_res is ref_out
    utils.gems_assert_close(
        out,
        ref_out,
        dtype,
        reduce_dim=_max_reduce_dim(shapes),
        atol=_atol_base(dtype),
    )
