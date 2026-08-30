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
from . import test_utils as tu  # noqa: E402

# ``_nnz`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._nnz`` and ``-m _nnz`` both work.
setattr(
    pytest.mark,
    "_nnz",
    MarkDecorator(Mark("_nnz", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_nnz(Tensor self) -> int reports the number of *stored* entries of a
# sparse tensor (Sparse* / SparseCsr* backends). It never inspects the index or
# value payload: explicit zeros, nan/inf, duplicate (uncoalesced) coordinates
# and fully-dense storage all count as stored entries, and the result is not
# the number of distinct coordinates. Dense tensors have no Sparse* dispatch
# for this operator (they raise NotImplementedError), so every workload below
# feeds a sparse COO or CSR tensor.
#
# Coverage:
#   * layouts: (shape, sparse_dim, nnz) cases from the quick/core/all shape
#     levels, ranks 1-7, all-sparse and hybrid sparse+dense;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the reported count is identical for all of them);
#   * edge cases: empty (nnz == 0, dense and hybrid), uncoalesced, explicit
#     zeros, fully-dense sparse, nan/inf/-0.0 values, and the SparseCsr layout
#     (2-D, batched 3-D and 2-D-with-dense-dims);
#   * negative: dense tensors and non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).

_NNZ_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_NNZ_INT_DTYPES = utils.ALL_INT_DTYPES
_NNZ_DTYPES = _NNZ_FLOAT_DTYPES + _NNZ_INT_DTYPES + utils.BOOL_TYPES

# (shape, sparse_dim, nnz) triples covering 1-D/2-D/3-D all-sparse, 2-D/3-D
# hybrid, and mixed sparse+dense ranks up to 5-D.
_NNZ_COO_CASES_CORE = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

# Higher-rank layouts for the "all"/"extended" TEST_LEVEL: 4-D all-sparse and
# hybrid ranks up to 7-D.
_NNZ_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4, 9),
    ((3, 6, 4, 4, 6, 5), 4, 11),
    ((7, 3, 12, 4, 2, 15), 5, 10),
    ((3, 4, 2, 5, 3, 4, 2), 3, 13),
]


def _coo_cases():
    """(shape, sparse_dim, nnz) layouts selected by the TEST_LEVEL env var."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    if tu.LEVEL in ("all", "extended"):
        return _NNZ_COO_CASES_CORE + _NNZ_COO_CASES_ALL
    return _NNZ_COO_CASES_CORE


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    if tu.LEVEL in ("all", "extended"):
        return [((3, 4), 2, 7), ((3, 4, 2), 2, 12), ((12, 9, 3, 6), 4, 9)]
    return [((3, 4), 2, 7), ((3, 4, 2), 2, 12)]


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper (tu.make_input) and the sparse tensor is created
    # on the test device. Duplicate indices are allowed and merely leave the
    # tensor uncoalesced (covered explicitly below).
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + dense_shape, value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_csr_input(shape, nnz, dtype, value_range, seed=0):
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
    values = tu.make_input(dtype, (nnz,), value_range)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so ``_nnz`` reports the per-batch stored count.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._nnz is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._nnz callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_nnz", getattr(flag_gems, "_nnz", None))


def _assert_result(res_out, ref_out, nnz):
    # _nnz returns a plain Python int, so exact equality is required and no
    # tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == nnz


@pytest.mark._nnz
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The reported count must be the number of stored entries.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)
    assert inp.sparse_dim() == sparse_dim


@pytest.mark._nnz
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_coo_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the reported count never changes because _nnz
    # reads only layout metadata.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_empty(dtype):
    # nnz == 0: empty indices and values; the reported count is 0.
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
def test__nnz_empty_hybrid(dtype):
    # nnz == 0 with dense dimensions: the hybrid layout is preserved and the
    # reported count stays 0.
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 0)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _nnz must report the 5
    # *stored* entries, not the 3 distinct coordinates (the (0, 1) coordinate
    # is stored three times).
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
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
    # Explicit zeros are stored entries: _nnz counts them (3, not 1), unlike
    # nnz-style queries that drop zero values.
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype == torch.bool:
        values = torch.tensor([False, True, False], dtype=dtype)
    elif dtype.is_floating_point:
        values = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    else:
        values = torch.tensor([0, 1, 0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 3)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_full_storage(dtype):
    # Fully-dense sparse storage: every logical position is stored, so the
    # reported count equals numel.
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    )
    indices = indices.reshape(2, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_FLOAT_DTYPES)
def test__nnz_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: all six entries count.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 6)


@pytest.mark._nnz
@pytest.mark.parametrize("case", [(4, 4), (2, 4, 4), (3, 5, 7)])
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr(case, dtype):
    # SparseCsr dispatch: 2-D stores nnz entries total; batched 3-D stores the
    # same nnz entries per batch (shared crow/col pattern), so _nnz reports the
    # per-batch stored count.
    shape = case
    nnz = 5 if len(shape) == 2 else 3
    inp = _make_csr_input(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)


@pytest.mark._nnz
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr_dense_dims(dtype):
    # CSR layout with dense dimensions: shape (rows, cols, dense); _nnz reports
    # the number of stored (row, col) blocks, independent of the dense block.
    rows, cols, dense, nnz = 4, 4, 3, 5
    # crow segments: row0 -> 1, row1 -> 1, row2 -> 2, row3 -> 1 stored block.
    crow = torch.tensor([0, 1, 2, 4, 5])
    col = torch.tensor([0, 1, 0, 1, 2])
    values = tu.make_input(dtype, (nnz, dense), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow, col, values, (rows, cols, dense), device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)


@pytest.mark._nnz
def test__nnz_dense_raises():
    # _nnz dispatches only on Sparse*/SparseCsr* backends; dense tensors have
    # no implementation and raise. The candidate must fail too rather than
    # silently report a bogus count.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._nnz(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._nnz
def test__nnz_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._nnz(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
