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
#   * shape levels: (shape, sparse_dim, nnz) layouts from the quick/core/all
#     levels, ranks 1-7, all-sparse and hybrid sparse+dense, with varying nnz
#     so the (sparse_dim, nnz) shape of the result is exercised;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the returned indices are identical for all of
#     them);
#   * edge cases: empty (nnz == 0, dense and hybrid), uncoalesced (duplicate,
#     unsorted coordinates), fully-dense sparse storage, and nan/inf/-0.0
#     values (all ignored by the accessor);
#   * negative: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a view
# of the input's own storage (there is nothing to broadcast against) and its
# result is an int64 metadata tensor (nothing to differentiate).

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse COO runtime supports: every float, int, and bool
# family.
_INDICES_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

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

# Higher-rank layouts for the "all"/"extended" TEST_LEVEL: 4-D all-sparse and
# hybrid ranks up to 7-D.
_INDICES_COO_CASES_ALL = [
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
        return _INDICES_COO_CASES_CORE + _INDICES_COO_CASES_ALL
    return _INDICES_COO_CASES_CORE


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
    # for every storage dtype (bool/int snap the range to the representable
    # set). The returned (sparse_dim, nnz) index view must match the reference
    # exactly and alias the input's indices storage.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
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
    # extreme and degenerate); the returned index view never changes because
    # _indices reads only layout metadata, not the values payload.
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
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


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
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
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
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
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
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._indices
def test__indices_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
