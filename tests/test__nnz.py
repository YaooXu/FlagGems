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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# The KernelGen harness runs pytest in-process with its own ``tests`` package
# (kernelgen/tests) earlier on sys.path than this checkout's ``tests`` package.
# With ``--import-mode=importlib`` pytest does not prepend the checkout root, so
# ``tests`` would resolve to the harness's package and ``from . import
# accuracy_utils`` would fail with ImportError during collection. Re-point the
# ``tests`` package at this file's directory before importing the helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests as _tests_pkg  # noqa: E402

if _HERE not in getattr(_tests_pkg, "__path__", []):
    sys.modules.pop("tests", None)
    import tests as _tests_pkg  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

# ``_nnz`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._nnz`` and ``-m _nnz`` both work.
setattr(
    pytest.mark,
    "_nnz",
    MarkDecorator(Mark("_nnz", (), {}, _ispytest=True), _ispytest=True),
)

_NNZ_CASES = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

_NNZ_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, sparse_dim, nnz, dtype, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    if dtype.is_floating_point:
        values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_csr_input(shape, nnz, dtype, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    col_indices = torch.randint(0, cols, (nnz,), dtype=torch.long, generator=gen)
    cuts = torch.sort(
        torch.randint(0, nnz + 1, (rows - 1,), dtype=torch.long, generator=gen)
    ).values
    crow_indices = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            cuts,
            torch.full((1,), nnz, dtype=torch.long),
        ]
    )
    if dtype.is_floating_point:
        values = torch.randn(nnz, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (nnz,), dtype=dtype, generator=gen)
    if len(shape) == 3:
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _resolve_gems_op():
    return flag_gems.testing.resolve_gems_op("_nnz", getattr(flag_gems, "_nnz", None))


def _assert_result(res_out, ref_out, nnz):
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == nnz


@pytest.mark._nnz
@pytest.mark.parametrize("case", _NNZ_CASES)
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)
    assert inp._nnz() == nnz
    assert inp.sparse_dim() == sparse_dim


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 0)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 5)
    assert inp.coalesce()._nnz() == 3


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_explicit_zeros(dtype):
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype.is_floating_point:
        values = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    elif dtype == torch.bool:
        values = torch.tensor([False, True, False])
    else:
        values = torch.tensor([0, 1, 0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 3)


@pytest.mark._nnz
@pytest.mark.parametrize("case", [(4, 4), (2, 4, 4), (3, 5, 7)])
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr(case, dtype):
    shape = case
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)
