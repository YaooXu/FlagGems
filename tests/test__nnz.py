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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

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
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * dtypes -- the full required set (int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/
#     bf16/fp16/int32/int64) plus fp64/int16/bool where the device supports
#     them, probed at import time with tu.supported_dtypes;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over the spec shape set and representative COO layouts;
#   * shapes/layouts -- the tu.selected_shapes() levels (quick: (2,19,7); full:
#     1-D .. 5-D) mapped to all-sparse COO, plus dedicated sparse layouts
#     (ranks 1-7, all-sparse and hybrid sparse+dense) and SparseCsr (2-D,
#     batched 3-D and CSR-with-dense-dims);
#   * edge cases -- empty (nnz == 0, dense and hybrid), uncoalesced, explicit
#     zeros, fully-dense sparse and nan/inf/-0.0 values;
#   * negative cases -- dense tensors and non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).

_NNZ_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)


def _sparse_dtype_probe(op_name, dtype):
    """Report whether ``op_name`` accepts a tiny sparse COO tensor of ``dtype``."""
    del op_name
    try:
        indices = torch.zeros(2, 1, dtype=torch.long, device=flag_gems.device)
        values = torch.zeros(1, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_coo_tensor(indices, values, (1, 1), device=flag_gems.device)
        return isinstance(torch.ops.aten._nnz(utils.to_reference(inp)), int)
    except Exception:
        return False


# Probe the device before parametrizing: an op/dtype pair that cannot run must
# not be turned into a red test.
_NNZ_DTYPES = tu.supported_dtypes(
    "_nnz", candidates=_NNZ_DTYPE_CANDIDATES, probe=_sparse_dtype_probe
) or [torch.float32]
_NNZ_FLOAT_DTYPES = [dtype for dtype in _NNZ_DTYPES if dtype.is_floating_point]
# float8 has no sparse coalesce kernel, so the coalesce-count assertion below
# is only checked for the non-fp8 storage dtypes.
_NNZ_COALESCE_DTYPES = [
    dtype
    for dtype in _NNZ_DTYPES
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
]

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

# Higher-rank layouts for the full level: 4-D all-sparse and hybrid ranks up to
# 7-D.
_NNZ_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4, 9),
    ((3, 6, 4, 4, 6, 5), 4, 11),
    ((7, 3, 12, 4, 2, 15), 5, 10),
    ((3, 4, 2, 5, 3, 4, 2), 3, 13),
]

# SparseCsr layouts: 2-D, batched 3-D (same crow/col pattern per batch).
_NNZ_CSR_CASES = [(4, 4), (2, 4, 4), (3, 5, 7)]

# Fixed stored-entry count for the spec-shape sweeps: small enough that every
# mapped shape stays cheap, and > 1 so duplicate (uncoalesced) coordinates are
# exercised for the small index spaces.
_NNZ_SPEC_NNZ = 6


def _coo_cases():
    """(shape, sparse_dim, nnz) layouts selected by the quick / full level."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    return _NNZ_COO_CASES_CORE + _NNZ_COO_CASES_ALL


def _spec_shapes(min_rank=1, max_rank=None):
    """``tu.selected_shapes()`` filtered to the ranks a sparse layout supports.

    The shared shape set is dense-only and includes a 0-dim entry, which has no
    sparse analogue; ``min_rank``/``max_rank`` keep only the ranks a COO /
    CSR layout can represent.
    """
    shapes = [shape for shape in tu.selected_shapes() if len(shape) >= min_rank]
    if max_rank is not None:
        shapes = [shape for shape in shapes if len(shape) <= max_rank]
    return shapes


def _make_values(dtype, shape, value_range):
    """Value-range helper with unsigned-bound snapping.

    ``tu.make_input`` cannot build a uint8 tensor for the ``[-1, 0]`` range
    (``-1`` is not representable); the op only counts stored entries, so the
    range is snapped to its representable subset for that one dtype/range pair.
    """
    if dtype == torch.uint8 and value_range == ["-1", "0"]:
        value_range = ["0", "0"]
    return tu.make_input(dtype, shape, value_range)


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper and the sparse tensor is created on the test
    # device. Duplicate indices are allowed and merely leave the tensor
    # uncoalesced (covered explicitly below).
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + dense_shape, value_range)
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
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so ``_nnz`` reports the per-batch stored count.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = _make_values(dtype, (shape[0], nnz), value_range)
    else:
        values = _make_values(dtype, (nnz,), value_range)
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
def test__nnz_coo_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every probed storage dtype. The reported count must be the number of
    # stored entries, independent of rank, sparsity pattern and value payload.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)
    assert inp.sparse_dim() == sparse_dim


@pytest.mark._nnz
@pytest.mark.parametrize("shape", _spec_shapes(min_rank=1))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_spec_shapes_value_ranges(shape, value_range, dtype):
    # Shape level from the shared spec set, mapped to all-sparse COO
    # (sparse_dim == ndim), crossed with the five spec value ranges. The
    # stored values never change the reported count, but they exercise the
    # value-range machinery end to end on every rank.
    nnz = _NNZ_SPEC_NNZ
    inp = _make_coo_input(shape, len(shape), nnz, dtype, value_range)
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
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 5)
    if dtype in _NNZ_COALESCE_DTYPES:
        # Distinct coordinates collapse to 3 values once coalesced.
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
    else:
        values = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
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
    values = _make_values(dtype, (nnz,), ["-1", "1"])
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
@pytest.mark.parametrize("case", _NNZ_CSR_CASES)
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
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_csr_value_ranges(value_range, dtype):
    # The CSR path is value-independent too: sweep the spec ranges through a
    # 2-D CSR tensor with a fixed 5-entry crow/col pattern.
    shape, nnz = (4, 4), 5
    inp = _make_csr_input(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nnz(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, nnz)


@pytest.mark._nnz
@pytest.mark.parametrize("shape", _spec_shapes(min_rank=2, max_rank=3))
@pytest.mark.parametrize("dtype", _NNZ_DTYPES)
def test__nnz_spec_shapes_csr(shape, dtype):
    # Shape level from the shared spec set on the other compressed dispatch
    # (SparseCsr): 2-D uses one shared crow/col pattern, batched 3-D stores the
    # same per-batch count.
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
    values = _make_values(dtype, (nnz, dense), ["-1", "1"])
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
    with pytest.raises(
        (NotImplementedError, RuntimeError, TypeError, ValueError, AttributeError)
    ):
        _resolve_gems_op()(inp)


@pytest.mark._nnz
def test__nnz_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar / None hits the
    # invalid combination of arguments path and raises (dense tensors are
    # rejected for a different reason, covered by the test above).
    with pytest.raises(RuntimeError):
        torch.ops.aten._nnz(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AttributeError)
    ):
        _resolve_gems_op()(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AttributeError)
    ):
        _resolve_gems_op()(None)
