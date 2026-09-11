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

# ``_indices`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._indices`` and ``-m _indices`` both
# work.
setattr(
    pytest.mark,
    "_indices",
    MarkDecorator(Mark("_indices", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_indices(Tensor(a) self) -> Tensor(a) returns the (sparse_dim, nnz)
# int64 index tensor of a sparse COO tensor as a *view* of the input's internal
# indices storage. It never depends on the stored values, and it dispatches only
# on the Sparse (COO) backend keys (dense and SparseCsr tensors raise
# NotImplementedError), so every workload below feeds a sparse COO tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * dtypes -- the full required set (int8/uint8/fp8_e4m3fn/fp8_e5m2/fp32/
#     bf16/fp16/int32/int64) plus fp64/int16/bool where the device supports
#     them, probed at import time with tu.supported_dtypes over a tiny sparse
#     COO tensor;
#   * value ranges -- tu.selected_ranges() ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) over both representative COO layouts and the spec shape levels;
#   * shapes/layouts -- the tu.selected_shapes() levels (quick: (2,19,7); full:
#     1-D .. 5-D) mapped to all-sparse COO, plus dedicated sparse layouts
#     (ranks 1-5, all-sparse and hybrid sparse+dense) with varying nnz so the
#     (sparse_dim, nnz) shape of the result is exercised;
#   * edge cases -- empty (nnz == 0, dense and hybrid), uncoalesced (duplicate,
#     unsorted coordinates), explicit zeros, fully-dense sparse storage, and
#     nan/inf/-0.0 values (all ignored by the accessor);
#   * negative cases -- dense tensors, SparseCsr tensors and non-tensor inputs
#     are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a view
# of the input's own storage (there is nothing to broadcast against) and its
# result is an int64 metadata tensor (nothing to differentiate).

_INDICES_DTYPE_CANDIDATES = list(
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
    """Report whether ``op_name`` accepts a tiny sparse COO tensor of ``dtype``.

    ``_indices`` takes a sparse tensor, so the default dense probe of
    ``tu.supported_dtypes`` cannot decide dtype support; this probe builds a
    real sparse COO input and calls the aten reference.
    """
    del op_name
    try:
        indices = torch.zeros(2, 1, dtype=torch.long, device=flag_gems.device)
        values = torch.zeros(1, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_coo_tensor(indices, values, (1, 1), device=flag_gems.device)
        out = torch.ops.aten._indices(utils.to_reference(inp))
        return out.dtype == torch.int64 and tuple(out.shape) == (2, 1)
    except Exception:
        return False


# Probe the device before parametrizing: an op/dtype pair that cannot run must
# not be turned into a red test. If the probe yields nothing, keep the full
# candidate list rather than a float32-only fallback, so a failed/absent probe
# never silently drops the spec-required int8/uint8/fp8 dtypes.
_INDICES_DTYPES = tu.supported_dtypes(
    "_indices", candidates=_INDICES_DTYPE_CANDIDATES, probe=_sparse_dtype_probe
) or list(_INDICES_DTYPE_CANDIDATES)
_INDICES_FLOAT_DTYPES = [dtype for dtype in _INDICES_DTYPES if dtype.is_floating_point]

# (shape, sparse_dim, nnz) triples covering 1-D/2-D/3-D all-sparse, 2-D/3-D
# hybrid, and mixed sparse+dense ranks up to 5-D.
_INDICES_COO_CASES_CORE = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

# Higher-rank layouts for the "all" level (no --quick): 4-D all-sparse and
# hybrid ranks up to 7-D.
_INDICES_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4, 9),
    ((3, 6, 4, 4, 6, 5), 4, 11),
    ((7, 3, 12, 4, 2, 15), 5, 10),
    ((3, 4, 2, 5, 3, 4, 2), 3, 13),
]

# Number of stored entries for the spec-shape sweep: small enough that every
# mapped shape stays cheap (sparse storage only materializes nnz entries), and
# > 1 so duplicate (uncoalesced) coordinates are exercised for small index
# spaces.
_INDICES_SPEC_NNZ = 6


def _coo_cases():
    """(shape, sparse_dim, nnz) layouts selected by pytest --quick vs default (full)."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    return _INDICES_COO_CASES_CORE + _INDICES_COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    return [((3, 4), 2, 7), ((3, 4, 2), 2, 12), ((12, 9, 3, 6), 4, 9)]


def _spec_shapes(min_rank=1, max_rank=None):
    """``tu.selected_shapes()`` filtered to the ranks a sparse layout supports.

    The shared shape set is dense-oriented and includes a 0-dim entry, which has
    no sparse analogue; ``min_rank``/``max_rank`` keep only the ranks a COO
    layout can represent.
    """
    shapes = [shape for shape in tu.selected_shapes() if len(shape) >= min_rank]
    if max_rank is not None:
        shapes = [shape for shape in shapes if len(shape) <= max_rank]
    return shapes


def _make_values(dtype, shape, value_range):
    """Value-range helper with unsigned-bound snapping.

    ``tu.make_input`` cannot build a uint8 tensor for the ``[-1, 0]`` range
    (``-1`` is not representable); the accessor only reads index metadata, so
    the range is snapped to its representable subset for that one dtype/range
    pair.
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


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._indices callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_indices", getattr(flag_gems, "_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # _indices returns a view of the input's internal (sparse_dim, nnz) int64
    # index tensor. The values are exact, and the schema annotation
    # Tensor(a) self -> Tensor(a) requires the result to alias the input's
    # indices storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert res_out.shape == (inp.sparse_dim(), inp._nnz())
    assert ref_out.shape == (ref_inp.sparse_dim(), ref_inp._nnz())
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal indices tensor (both on the candidate and the reference).
    assert res_out.data_ptr() == inp._indices().data_ptr()
    assert ref_out.data_ptr() == ref_inp._indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set), so its indices and values
    # still match the (untouched) input storage after the calls. Values may
    # legitimately hold nan/inf, so compare them with equal_nan for float
    # storage.
    utils.gems_assert_equal(inp._indices(), ref_inp._indices())
    if inp.dtype.is_floating_point:
        utils.gems_assert_equal(inp._values(), ref_inp._values(), equal_nan=True)
    else:
        utils.gems_assert_equal(inp._values(), ref_inp._values())


@pytest.mark._indices
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every probed storage dtype (bool/int snap the range to the
    # representable set). The returned (sparse_dim, nnz) index view must match
    # the reference exactly and alias the input's indices storage.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("shape", _spec_shapes(min_rank=1))
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_spec_shapes_value_ranges(shape, value_range, dtype):
    # Shape level from the shared spec set, mapped to all-sparse COO
    # (sparse_dim == ndim), crossed with the five spec value ranges. The stored
    # values never change the returned index view, but they exercise the
    # value-range machinery end to end on every rank.
    nnz = _INDICES_SPEC_NNZ
    inp = _make_coo_input(shape, len(shape), nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate) on representative all-sparse and hybrid layouts;
    # the returned index view never changes because _indices reads only layout
    # metadata, not the values payload.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty(dtype):
    # nnz == 0: indices and values are empty, but _indices must still return a
    # (sparse_dim, 0) int64 tensor (not a dense or wrongly-shaped tensor).
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_empty_hybrid(dtype):
    # nnz == 0 with dense dimensions: the returned indices tensor has shape
    # (2, 0), preserving the sparse_dim of the hybrid sparse layout.
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _indices must still
    # return exactly the stored index tensor (never a coalesced/sorted copy).
    # The (0, 1) coordinate is repeated three times and the entries are NOT
    # sorted, so a coalescing implementation would visibly change the result.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_explicit_zeros(dtype):
    # Explicit zeros are ordinary stored entries: the accessor returns their
    # coordinates too (all three), never dropping them like a value-based nnz
    # filter would.
    shape = (3, 3)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    if dtype == torch.bool:
        values = torch.tensor([False, False, False], dtype=dtype)
    else:
        values = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
    assert res_out.shape == (2, 3)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_DTYPES)
def test__indices_full_storage(dtype):
    # Fully-dense sparse storage: every logical position is stored, so the
    # (sparse_dim, numel) index tensor lists every coordinate exactly once.
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = _make_values(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", _INDICES_FLOAT_DTYPES)
def test__indices_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: _indices must still return
    # exactly the stored index tensor, unchanged, for every one of them.
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
def test__indices_dense_raises():
    # _indices dispatches only on the Sparse (COO) backend keys; dense tensors
    # have no implementation and raise. The candidate must fail too rather than
    # silently return a bogus index tensor.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(utils.to_reference(inp))
    with pytest.raises(
        (NotImplementedError, RuntimeError, TypeError, ValueError, AttributeError)
    ):
        _resolve_gems_op()(inp)


@pytest.mark._indices
def test__indices_csr_raises():
    # SparseCsr is a distinct dispatch key from Sparse (COO); _indices has no
    # SparseCsr implementation and raises. The candidate must reject it too.
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = tu.make_input(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 4), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._indices(utils.to_reference(inp))
    with pytest.raises(
        (NotImplementedError, RuntimeError, TypeError, ValueError, AttributeError)
    ):
        _resolve_gems_op()(inp)


@pytest.mark._indices
def test__indices_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._indices(3.14)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AttributeError)
    ):
        _resolve_gems_op()(3.14)
