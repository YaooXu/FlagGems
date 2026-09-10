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
from . import test_utils as tu

# aten::dim(Tensor self) -> int returns the number of dimensions of a tensor,
# i.e. ``len(self.size())``, for every layout the runtime supports: strided
# (dense, including non-contiguous / transposed views and empty tensors), sparse
# COO (all-sparse and hybrid) and sparse CSR (2-D and batched 3-D). It is a pure
# metadata query whose result never depends on the stored values or the storage
# dtype, so every workload below covers a distinct (shape, layout) pair. The
# result is a plain Python ``int``, so each workload asserts exact equality.
#
# Coverage (regular-operator spec, metadata adaptation):
#   * dtypes -- int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/bf16/fp16/int32/int64 plus
#     fp64/int16/bool where the device supports them, probed at import time
#     with tu.supported_dtypes;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over the spec shape set and representative dense / sparse COO /
#     sparse CSR layouts;
#   * shapes -- the tu.selected_shapes() levels (quick: (2,19,7); full: 0-dim
#     .. 5-dim) plus dedicated dense ranks 0-8, empty tensors, COO ranks 1-7
#     (all-sparse and hybrid) and CSR 2-D / batched 3-D;
#   * edge cases -- empty dense/COO/CSR, uncoalesced COO, non-contiguous
#     (transposed) dense views, and nan/inf/-inf/±0.0 stored values;
#   * negative cases -- non-tensor inputs are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against, and an int result has
# no autograd graph).

_DIM_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        [
            *tu.REQUIRED_DTYPES,  # int8, uint8, fp8_e4m3fn/e5m2, fp32, bf16, fp16, int32, int64
            *utils.ALL_FLOAT_DTYPES,  # + float64 where supported
            *utils.ALL_INT_DTYPES,  # + int16 where supported
            *utils.BOOL_TYPES,
        ]
    )
)

# Probe the device before parametrizing: an op/dtype pair that cannot run must
# not be turned into a red test.
_DIM_DTYPES = tu.supported_dtypes("dim", candidates=_DIM_DTYPE_CANDIDATES) or [
    torch.float32
]
_DIM_FLOAT_DTYPES = [dtype for dtype in _DIM_DTYPES if dtype.is_floating_point]

# Dense (strided) tensors: dim == len(shape). Ranks 0 through 5 cover the full
# range, including the degenerate scalar case (rank 0).
_DENSE_CASES_CORE = [
    ((), 0),
    ((5,), 1),
    ((3, 4), 2),
    ((8, 8, 8), 3),
    ((3, 4, 2, 5), 4),
    ((3, 4, 5, 4, 5), 5),
]

# Higher-rank strided tensors for the "all" level (default, no --quick).
_DENSE_CASES_ALL = [
    ((3, 6, 4, 4, 6, 5, 4), 7),
    ((7, 3, 12, 4, 2, 15, 2, 2), 8),
]

# Empty dense tensors: numel == 0, but the rank is still reported exactly.
_EMPTY_DENSE_CASES = [
    ((0,), 1),
    ((0, 5), 2),
    ((2, 0, 3), 3),
]

# Sparse COO tensors: (sparse_shape, dense_shape, nnz) with logical size
# ``sparse_shape + dense_shape`` and expected result
# ``len(sparse_shape) + len(dense_shape)``. Covers all-sparse layouts as well
# as mixed sparse+dense ranks from 1 up to 5.
_COO_CASES_CORE = [
    ((4, 4), (), 8),
    ((8, 8, 8), (), 64),
    ((4, 4), (3,), 8),
    ((2, 3, 4), (5,), 12),
    ((16, 16), (7, 13), 40),
    ((2, 3, 4), (5, 6), 12),
    ((3,), (4, 5, 6), 2),
]

# Higher-rank hybrid layouts for the "all" level (default, no --quick).
_COO_CASES_ALL = [
    ((12, 9, 3, 6), (4,), 9),
    ((3, 4, 2, 5, 3), (4, 2), 11),
]

# Sparse CSR tensors: (shape, nnz). dim is the full logical rank for both 2-D
# and batched layouts.
_CSR_CASES_CORE = [
    ((4, 4), 3),
    ((2, 4, 4), 5),
    ((3, 5, 7), 3),
]

# Additional batched CSR layout for the "all" level (default, no --quick).
_CSR_CASES_ALL = [
    ((3, 4, 4), 4),
]

# Empty CSR tensors: nnz == 0, plain 2-D and batched 3-D layouts.
_EMPTY_CSR_CASES = [
    (4, 4),
    (3, 4, 4),
]


def _dense_cases():
    """(shape, expected) strided layouts selected by --quick (quick) vs default."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    if tu.LEVEL == "all":
        return _DENSE_CASES_CORE + _DENSE_CASES_ALL


def _coo_cases():
    """(sparse_shape, dense_shape, nnz) COO layouts selected by --quick."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    if tu.LEVEL == "all":
        return _COO_CASES_CORE + _COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid COO layouts for the range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), (), 8)]
    if tu.LEVEL == "all":
        return [((3, 4), (), 7), ((3, 4), (3,), 8), ((12, 9, 3, 6), (4,), 9)]


def _csr_cases():
    """(shape, nnz) CSR layouts selected by --quick (quick) vs default."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    if tu.LEVEL == "all":
        return _CSR_CASES_CORE + _CSR_CASES_ALL


def _csr_value_range_cases():
    """Representative 2-D + batched CSR layouts for the range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 3)]
    if tu.LEVEL == "all":
        return [((4, 4), 3), ((2, 4, 4), 5)]


def _make_values(dtype, shape, value_range):
    """Value-range helper with unsigned-bound clamping.

    ``tu.make_input`` resolves the range symbols per dtype, but a negative low
    bound is not representable for ``uint8`` (``[-1, 0]`` collapses to an empty
    interval and ``torch.testing.make_tensor`` raises). ``dim`` never inspects
    the stored values, so the range is clamped to the representable subset for
    that one dtype/range pair.
    """
    if dtype == torch.uint8:
        low = max(0, int(tu.resolve_bound(value_range[0], dtype)))
        high = max(0, int(tu.resolve_bound(value_range[1], dtype)))
        if low > high:
            low = high
        if low == high:
            return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
        return torch.testing.make_tensor(
            shape, dtype=dtype, device=flag_gems.device, low=low, high=high
        )
    return tu.make_input(dtype, shape, value_range)


def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper and the sparse tensor is created on the test
    # device. Duplicate indices are allowed (the layout is simply uncoalesced),
    # which is covered explicitly below.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + tuple(dense_shape), value_range)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_csr(shape, nnz, dtype, value_range, seed=0):
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
    values = _make_values(dtype, (nnz,), value_range)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the logical rank is 3.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = values.expand(shape[0], -1).contiguous()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build a CSR tensor with nnz == 0: the rank of the layout is still
    reported exactly as for a populated tensor."""
    if len(shape) == 2:
        rows, _ = shape
        crow_indices = torch.zeros(rows + 1, dtype=torch.long)
        col_indices = torch.empty(0, dtype=torch.long)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        _, rows, _ = shape
        crow_indices = torch.zeros(shape[0], rows + 1, dtype=torch.long)
        col_indices = torch.empty(shape[0], 0, dtype=torch.long)
        values = torch.empty(shape[0], 0, dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected for this run wins. The default stays None
    # until flag_gems.dim is registered; resolution order is: (1) override,
    # (2) the direct flag_gems.dim callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("dim", getattr(flag_gems, "dim", None))


def _assert_result(res_out, ref_out, expected):
    # dim returns a plain Python int holding the rank, so exact equality is
    # required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == expected


@pytest.mark.dim
@pytest.mark.parametrize("shape, expected", _dense_cases())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_dense_layouts(shape, expected, dtype):
    # Values from [-1, 1]: negative and positive stored values for every probed
    # dtype; the reported rank depends only on the layout.
    inp = _make_values(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.dim
@pytest.mark.parametrize("shape, expected", _EMPTY_DENSE_CASES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_dense(shape, expected, dtype):
    # numel == 0, but the rank is still reported exactly.
    inp = _make_values(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, expected)


@pytest.mark.dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_dense_value_ranges(shape, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the reported rank never changes because dim
    # reads only layout metadata.
    inp = _make_values(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))


@pytest.mark.dim
@pytest.mark.parametrize(
    "shape",
    [(16, 32), (8, 16, 32), (4, 8, 16, 32)] if tu.LEVEL == "all" else [(2, 19, 7)],
)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_noncontiguous_dense(shape, dtype):
    # Transposed (non-contiguous) views: dim reads only the metadata, so the
    # reported rank is unchanged by the memory layout of the view.
    inp = _make_values(dtype, shape, ["-1", "1"]).transpose(0, -1)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))


@pytest.mark.dim
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


@pytest.mark.dim
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_coo_value_ranges(case, value_range, dtype):
    # Sparse COO path is value-independent: sweep the five spec ranges through
    # all-sparse and hybrid layouts.
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))


@pytest.mark.dim
@pytest.mark.parametrize("case", _csr_cases())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_csr_layouts(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == 0
    assert inp.sparse_dim() == 2


@pytest.mark.dim
@pytest.mark.parametrize("case", _csr_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_csr_value_ranges(case, value_range, dtype):
    # Sparse CSR path is value-independent too: 2-D and batched 3-D sweeps.
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_coo(dtype):
    # nnz == 0: indices and values are empty, but the rank of the layout is
    # still reported exactly as for a populated tensor.
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))


@pytest.mark.dim
@pytest.mark.parametrize("shape", _EMPTY_CSR_CASES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_csr(shape, dtype):
    # nnz == 0: indices and values are empty, but the rank of the layout is
    # still reported exactly as for a populated tensor.
    inp = _make_empty_csr(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_uncoalesced_coo(dtype):
    # The (0, 0) coordinate is repeated, so the tensor is uncoalesced; dim must
    # still report the same rank as the coalesced form because it never
    # inspects the index or data values.
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = _make_values(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(sparse_shape) + len(dense_shape))


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_FLOAT_DTYPES)
def test_dim_nan_inf_dense(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values for a metadata query: the
    # strided path still reports len(shape).
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    ).reshape(2, 3)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_FLOAT_DTYPES)
def test_dim_nan_inf_coo(dtype):
    # The same values stored sparsely: dim reports the full rank (1 for this
    # 1-D layout) regardless of the nan/inf payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 1)


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_FLOAT_DTYPES)
def test_dim_nan_inf_csr(dtype):
    # The same values stored in CSR form: dim reports the full logical rank of
    # the layout (2) regardless of the nan/inf payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 2)


@pytest.mark.dim
def test_dim_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises. The candidate must fail too
    # rather than silently report a bogus rank.
    with pytest.raises(RuntimeError):
        torch.ops.aten.dim(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, AttributeError, NotImplementedError)
    ):
        _resolve_gems_op()(3.14)
