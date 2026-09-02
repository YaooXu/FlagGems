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

# ``_dimI`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimI`` and ``-m _dimI`` both work.
setattr(
    pytest.mark,
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimI(Tensor self) -> int reports the number of sparse dimensions of a
# sparse tensor (``sparse_dim``): a pure metadata query whose result never
# depends on the index/data values or the storage dtype. The Sparse* backends
# are the only dispatch targets (dense and SparseCsr* tensors raise
# NotImplementedError), so every workload below feeds a sparse COO tensor.
#
# Coverage:
#   * layouts: (shape, sparse_dim) cases from the quick/all shape levels,
#     ranks 1-5, all-sparse and hybrid sparse+dense;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the reported sparse dim is identical for all of
#     them);
#   * edge cases: empty (nnz == 0, dense and hybrid), uncoalesced, and nan/inf
#     values (all ignored by the metadata query);
#   * negative: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).

_DIMI_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# (shape, sparse_dim) pairs covering 1-D/2-D/3-D all-sparse, 2-D/3-D hybrid,
# and mixed sparse+dense ranks up to 5-D.
_DIMI_COO_CASES_CORE = [
    ((5,), 1),
    ((3, 4), 2),
    ((3, 4), 1),
    ((8, 8, 8), 3),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 1),
    ((3, 4, 5, 4, 5), 3),
]

# Higher-rank layouts for the "all" level (no --quick): 4-D all-sparse and
# hybrid ranks up to 7-D.
_DIMI_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4),
    ((3, 6, 4, 4, 6, 5), 4),
    ((7, 3, 12, 4, 2, 15), 5),
    ((3, 4, 2, 5, 3, 4, 2), 3),
]


def _coo_cases():
    """(shape, sparse_dim) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2)]
    if tu.LEVEL == "all":
        return _DIMI_COO_CASES_CORE + _DIMI_COO_CASES_ALL


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2)]
    if tu.LEVEL == "all":
        return [((3, 4), 2), ((3, 4, 2), 2), ((12, 9, 3, 6), 4)]


def _make_coo_input(shape, sparse_dim, dtype, value_range, nnz=8, seed=0):
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
    # default stays None until flag_gems._dimI is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimI callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op("_dimI", getattr(flag_gems, "_dimI", None))


def _assert_result(res_out, ref_out, sparse_dim):
    # _dimI returns a plain Python int holding the sparse dimension count, so
    # exact equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The reported count must be the layout's sparse dim.
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == sparse_dim
    assert inp.dense_dim() == len(shape) - sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_coo_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the reported sparse dim never changes because
    # _dimI reads only layout metadata.
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty(dtype):
    # nnz == 0: indices and values are empty, but the sparse dims of the layout
    # are still reported exactly as for a populated tensor.
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty_hybrid(dtype):
    # nnz == 0 with dense dimensions: the hybrid layout is preserved and the
    # sparse dims stay exactly as for a populated tensor.
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _dimI must still report
    # the same sparse dim as the coalesced form (it never inspects the index or
    # data values). The (0, 1) coordinate is repeated three times.
    shape, sparse_dim = (3, 4), 2
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__dimI_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: the metadata query still
    # reports the sparse dim of the layout, independent of the payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 1)


@pytest.mark._dimI
def test__dimI_dense_raises():
    # _dimI dispatches only on the sparse COO backends; dense tensors have no
    # implementation and raise. The candidate must fail too rather than
    # silently report a bogus count.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._dimI
def test__dimI_csr_raises():
    # SparseCsr* backends have no kernel for _dimI (unlike _nnz, which does
    # dispatch there): a CSR tensor raises instead of reporting its 2 sparse
    # dims, and the candidate must fail too.
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._dimI
def test__dimI_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimI(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
