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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimV(Tensor self) -> int reports the number of dense dimensions of a
# sparse tensor (``dense_dim``): a pure metadata query whose result never
# depends on the index/data values or the storage dtype. The Sparse* backends
# are the only dispatch targets (dense and SparseCsr* tensors raise
# NotImplementedError), so every workload below feeds a sparse COO tensor.
#
# Coverage:
#   * layouts: (shape, dense_dim) cases from the quick/core/all shape levels,
#     ranks 1-7, all-sparse and hybrid sparse+dense;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (the reported dense dim is identical for all of
#     them);
#   * edge cases: empty (nnz == 0, dense and hybrid), uncoalesced, and nan/inf
#     values (all ignored by the metadata query);
#   * negative: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain Python int (there is nothing to broadcast against or differentiate).

_DIMV_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# (shape, dense_dim) pairs covering 1-D/2-D/3-D all-sparse (dense_dim == 0),
# 2-D/3-D hybrid, and mixed sparse+dense ranks up to 5-D. dense_dim is the
# reported result and sparse_dim is derived as ``len(shape) - dense_dim``.
_DIMV_COO_CASES_CORE = [
    ((5,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
    ((8, 8, 8), 0),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 3),
    ((3, 4, 5, 4, 5), 2),
]

# Higher-rank layouts for the "all"/"extended" TEST_LEVEL: 4-D all-sparse and
# hybrid ranks up to 7-D.
_DIMV_COO_CASES_ALL = [
    ((12, 9, 3, 6), 0),
    ((3, 6, 4, 4, 6, 5), 2),
    ((7, 3, 12, 4, 2, 15), 3),
    ((3, 4, 2, 5, 3, 4, 2), 4),
]


def _dimv_cases():
    """(shape, dense_dim) layouts selected by the TEST_LEVEL env var."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 1)]
    if tu.LEVEL in ("all", "extended"):
        return _DIMV_COO_CASES_CORE + _DIMV_COO_CASES_ALL
    return _DIMV_COO_CASES_CORE


def _dimv_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 1)]
    if tu.LEVEL in ("all", "extended"):
        return [((3, 4), 1), ((3, 4, 2), 2), ((12, 9, 3, 6), 0)]
    return [((3, 4), 1), ((3, 4, 2), 2)]


def _make_coo_input(shape, dense_dim, dtype, value_range, nnz=8, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper (tu.make_input) and the sparse tensor is created
    # on the test device. Duplicate indices are allowed and merely leave the
    # tensor uncoalesced (covered explicitly below).
    sparse_dim = len(shape) - dense_dim
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
    # default stays None until flag_gems._dimV is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimV callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op("_dimV", getattr(flag_gems, "_dimV", None))


def _assert_result(res_out, ref_out, dense_dim):
    # _dimV returns a plain Python int holding the dense dimension count, so
    # exact equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == dense_dim


@pytest.mark._dimV
@pytest.mark.parametrize("case", _dimv_cases())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The reported count must be the layout's dense dim.
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() == len(shape) - dense_dim


@pytest.mark._dimV
@pytest.mark.parametrize("case", _dimv_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_coo_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the reported dense dim never changes because
    # _dimV reads only layout metadata.
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty(dtype):
    # nnz == 0: indices and values are empty, but the dense dims of the layout
    # are still reported exactly as for a populated tensor.
    shape, dense_dim = (3, 4), 0
    sparse_dim = len(shape) - dense_dim
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty_hybrid(dtype):
    # nnz == 0 with dense dimensions: the hybrid layout is preserved and the
    # dense dims stay exactly as for a populated tensor.
    shape, dense_dim = (4, 5, 6), 2
    sparse_dim = len(shape) - dense_dim
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 5, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _dimV must still report
    # the same dense dim as the coalesced form (it never inspects the index or
    # data values). The (0, 0) coordinate is repeated.
    shape, dense_dim = (2, 2, 3), 1
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = tu.make_input(dtype, (5, 3), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__dimV_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: the metadata query still
    # reports the dense dim of the layout, independent of the payload.
    values = torch.tensor(
        [
            [float("nan"), float("inf")],
            [float("inf"), float("-inf")],
            [0.0, -0.0],
            [1.5, 2.5],
            [float("nan"), 1.0],
            [float("-inf"), 0.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6, 2), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 1)


@pytest.mark._dimV
def test__dimV_dense_raises():
    # _dimV dispatches only on the sparse COO backends; dense tensors have no
    # implementation and raise. The candidate must fail too rather than
    # silently report a bogus count.
    inp = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._dimV
def test__dimV_csr_raises():
    # SparseCsr* backends have no kernel for _dimV: a CSR tensor raises instead
    # of reporting its 0 dense dims, and the candidate must fail too.
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = tu.make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(utils.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._dimV
def test__dimV_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimV(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
