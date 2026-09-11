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

# ``_values`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the marker
# directly on the MarkGenerator so ``@pytest.mark._values`` and ``-m _values``
# both work.
setattr(
    pytest.mark,
    "_values",
    MarkDecorator(Mark("_values", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_values(Tensor(a) self) -> Tensor(a) returns the (nnz,) + dense_shape
# values tensor of a sparse COO tensor as a VIEW that aliases the input's
# internal values storage (the schema annotation Tensor(a) -> Tensor(a) forces
# the alias). The entries are exactly the stored values, in storage order,
# independent of the stored indices; no coalescing or filtering of explicit
# zeros happens. Only the Sparse* (COO) backends have a kernel: dense and
# SparseCsr tensors raise NotImplementedError.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * dtype grid: every storage dtype the sparse-COO runtime and the aten
#     reference accept (int8/uint8/fp8-e4m3fn/fp8-e5m2/fp16/fp32/bf16, plus
#     fp64/int16/int32/int64/bool when supported), probed on a real sparse input
#     through tu.supported_dtypes so a backend lacking e.g. fp8 drops that dtype
#     instead of failing;
#   * shape levels: (shape, sparse_dim, nnz) layouts from the quick/all levels,
#     ranks 1-7, all-sparse and hybrid sparse+dense, with varying nnz so the
#     (nnz,) + dense_shape shape of the result is exercised;
#   * value ranges: the five spec ranges ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) via tu.make_input over representative layouts, so negative,
#     positive, dtype-extreme and degenerate values are all returned verbatim;
#   * boundaries: empty (nnz == 0, dense and hybrid), fully-stored
#     (nnz == prod(shape)), uncoalesced (duplicate + unsorted coordinates), and
#     nan/inf/-inf/+-0 values (all preserved verbatim);
#   * negatives: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected by both the reference and the candidate.
#
# No broadcast/backward dimensions apply: the operator is unary, returns an
# alias of the input's own storage (there is nothing to broadcast against), and
# its result is a non-differentiable metadata tensor (nothing to differentiate).

# ---------------------------------------------------------------------------
# Dtype support (probed on a real sparse COO input)
# ---------------------------------------------------------------------------

# Required spec dtypes first, followed by the shared float/int/bool families.
_VALUES_DTYPE_CANDIDATES = list(
    dict.fromkeys(
        tu.REQUIRED_DTYPES
        + utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + utils.BOOL_TYPES
    )
)


def _probe_values_sparse(operator, dtype):
    """Return True when ``aten::<operator>`` accepts a sparse COO tensor of dtype.

    The generic dense probe in ``tu.supported_dtypes`` cannot be used here:
    ``_values`` is a Sparse-only operator, so a dense input always raises. Build
    the smallest real sparse COO input instead and call the reference op.
    """
    try:
        indices = torch.zeros(1, 1, dtype=torch.long, device=flag_gems.device)
        values = torch.ones(1, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_coo_tensor(indices, values, (2,), device=flag_gems.device)
        getattr(torch.ops.aten, operator).default(inp)
        return True
    except Exception:
        return False


# If the probe yields nothing, keep the full candidate list rather than a
# float32-only fallback, so a failed/absent probe never silently drops the
# spec-required int8/uint8/fp8 dtypes.
_VALUES_DTYPES = tu.supported_dtypes(
    "_values", candidates=_VALUES_DTYPE_CANDIDATES, probe=_probe_values_sparse
) or list(_VALUES_DTYPE_CANDIDATES)

_VALUES_FLOAT_DTYPES = [d for d in _VALUES_DTYPES if d.is_floating_point]

# ---------------------------------------------------------------------------
# Sparse COO layouts: (shape, sparse_dim, nnz)
# ---------------------------------------------------------------------------

# 1-D/2-D/3-D all-sparse, 2-D/3-D hybrid, and mixed sparse+dense ranks up to
# 5-D.
_VALUES_COO_CASES_CORE = [
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
_VALUES_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4, 48),
    ((3, 6, 4, 4, 6, 5), 4, 64),
    ((7, 3, 12, 4, 2, 15), 5, 80),
    ((3, 4, 2, 5, 3, 4, 2), 3, 96),
]

# Quick level still spans all-sparse 1-D/2-D/3-D plus hybrid, so the quick
# dtype x range x layout grid clears tu.MIN_CASES as well.
_VALUES_COO_CASES_QUICK = [
    ((2, 19, 7), 2, 8),
    ((3, 4), 2, 7),
    ((3, 4, 2), 2, 12),
]


def _coo_cases():
    """(shape, sparse_dim, nnz) layouts selected by pytest --quick vs default."""
    if tu.LEVEL == "quick":
        return _VALUES_COO_CASES_QUICK
    return _VALUES_COO_CASES_CORE + _VALUES_COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return _VALUES_COO_CASES_QUICK
    return [((3, 4), 2, 7), ((3, 4, 2), 2, 12), ((12, 9, 3, 6), 4, 48)]


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper (which clamps a negative bound into the dtype's
    # range, realizing ``["-1", "0"]`` on uint8 as a constant zero fill) and the
    # sparse tensor is created on the test device. Duplicate indices merely leave
    # the tensor uncoalesced (covered explicitly below).
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


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._values is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._values callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_values", getattr(flag_gems, "_values", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # _values returns a view of the input's internal (nnz,) + dense_shape
    # values tensor. The stored entries come back verbatim (no coalescing, no
    # filtering of explicit zeros) with the storage dtype preserved, and the
    # schema annotation Tensor(a) self -> Tensor(a) requires the result to
    # alias the input's values storage.
    assert res_out.dtype == ref_out.dtype == inp.dtype
    assert res_out.shape == ref_out.shape == inp._values().shape
    assert ref_out.shape == ref_inp._values().shape
    # The view must preserve the stored values bit-for-bit (nan/inf included),
    # so exact equality is required for every dtype.
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal values tensor.
    assert res_out.data_ptr() == inp._values().data_ptr()
    assert ref_out.data_ptr() == ref_inp._values().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot,
    # so its values still match the (untouched) input storage after the calls.
    utils.gems_assert_equal(inp._values(), ref_inp._values(), equal_nan=True)


# ---------------------------------------------------------------------------
# Correctness workloads
# ---------------------------------------------------------------------------


@pytest.mark._values
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_layouts(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The returned view must preserve them verbatim for every layout.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); _values must return them verbatim, unchanged and
    # still aliased to the input's values storage.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_empty(dtype):
    # nnz == 0: indices and values are empty, but _values must still return a
    # (0,) + dense_shape tensor with the storage dtype (not a dense or
    # wrongly-shaped tensor).
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_empty_hybrid(dtype):
    # nnz == 0 with dense dimensions: the returned values tensor has shape
    # (0, 6), preserving the dense block shape of the hybrid sparse layout.
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_full_storage(dtype):
    # Fully-stored sparse COO (nnz == prod(shape), every coordinate unique and
    # coalesced): _values must return the whole stored values tensor, in
    # storage order, not a filtered subset.
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _values must still return
    # exactly the stored values tensor (never a coalesced/sorted copy) and stay
    # an alias of the input's values storage. The (0, 1) coordinate is repeated
    # three times and the entries are NOT sorted, so a coalescing
    # implementation would visibly change the result.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_FLOAT_DTYPES)
def test__values_nan_inf(dtype):
    # nan/inf/-inf/+-0.0 are ordinary stored values: _values must return them
    # verbatim (equal_nan=True), never sanitized. fp8-e4m3fn has no infinity
    # encoding, so inf/-inf collapse to nan there (still returned verbatim).
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark._values
def test__values_dense_raises():
    # _values dispatches only on the sparse COO backends; dense tensors have no
    # implementation and raise. The candidate must fail too rather than
    # silently returning a bogus tensor.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._values
def test__values_csr_raises():
    # SparseCsr is a distinct dispatch key from Sparse (COO): a CSR tensor
    # raises instead of returning its storage values, and the candidate must
    # fail too.
    crow_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([0, 1], dtype=torch.long)
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._values
def test__values_rejects_non_tensor():
    # The aten schema requires a Tensor; a non-tensor argument hits the
    # invalid-combination-of-arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._values(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten._values("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()("not-a-tensor")
