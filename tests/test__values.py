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

# The KernelGen verification harness stages this file in a temporary tree and
# runs pytest in-process with --import-mode=importlib, where the checkout root
# is not on sys.path (a `python -m pytest` invocation would normally place it
# there). Insert the checkout root so the sibling accuracy_utils package below
# resolves identically in the in-tree and staged verification layouts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

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
# the alias). The returned entries are exactly the stored values, in storage
# order, independent of the stored indices; no coalescing or filtering of
# explicit zeros happens. The Sparse* backends are the only dispatch targets
# (dense and SparseCsr* tensors raise NotImplementedError), so every workload
# below feeds a sparse COO tensor.
#
# Coverage:
#   * layouts: (shape, sparse_dim, nnz) cases selected from the quick/core/all
#     shape levels, ranks 1-7, all-sparse and hybrid sparse+dense, with varying
#     nnz so the (nnz,) + dense_shape shape of the result is exercised;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme and
#     degenerate value ranges (all returned verbatim);
#   * edge cases: empty (nnz == 0, dense and hybrid), uncoalesced (duplicate,
#     unsorted indices), and nan/inf/-inf/+-0 values (all preserved verbatim);
#   * negative: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain alias of the input's storage (there is nothing to broadcast against or
# differentiate).

_VALUES_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# (shape, sparse_dim, nnz) triples covering 1-D/2-D/3-D all-sparse, 2-D/3-D
# hybrid, and mixed sparse+dense ranks up to 5-D.
_VALUES_COO_CASES_CORE = [
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
_VALUES_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4, 48),
    ((3, 6, 4, 4, 6, 5), 4, 64),
    ((7, 3, 12, 4, 2, 15), 5, 80),
    ((3, 4, 2, 5, 3, 4, 2), 3, 96),
]


def _coo_cases():
    """(shape, sparse_dim, nnz) layouts selected by the TEST_LEVEL env var."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    if tu.LEVEL in ("all", "extended"):
        return _VALUES_COO_CASES_CORE + _VALUES_COO_CASES_ALL
    return _VALUES_COO_CASES_CORE


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    if tu.LEVEL in ("all", "extended"):
        return [((3, 4), 2, 7), ((3, 4, 2), 2, 12), ((12, 9, 3, 6), 4, 48)]
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
    # default stays None until flag_gems._values is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._values callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_values", getattr(flag_gems, "_values", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # _values returns a view of the input's internal (nnz,) + dense_shape
    # values tensor. The stored entries are returned verbatim (no coalescing,
    # no filtering of explicit zeros) with the storage dtype preserved, and the
    # schema annotation Tensor(a) self -> Tensor(a) requires the result to
    # alias the input's values storage.
    assert res_out.dtype == ref_out.dtype == inp.dtype
    assert res_out.shape == ref_out.shape == inp._values().shape
    # The view must preserve the stored values bit-for-bit (nan/inf included),
    # so exact equality is required for every dtype.
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal values tensor.
    assert res_out.data_ptr() == inp._values().data_ptr()
    assert ref_out.data_ptr() == ref_inp._values().data_ptr()
    # The accessor must not mutate the input: the result still matches the
    # (untouched) values captured on the reference copy before the call.
    utils.gems_assert_equal(res_out, ref_inp._values(), equal_nan=True)


@pytest.mark._values
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). The returned view must preserve them verbatim for every layout.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_coo_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); _values must return them verbatim, unchanged and
    # still aliased to the input's values storage.
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = utils.to_reference(inp)

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
    ref_inp = utils.to_reference(inp)

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
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", _VALUES_DTYPES)
def test__values_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _values must still return
    # exactly the stored values tensor (never a coalesced/sorted copy) and it
    # must stay an alias of the input's values storage. The (0, 1) coordinate
    # is repeated three times and the entries are NOT sorted, so a coalescing
    # implementation would visibly change the result.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark._values
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__values_nan_inf(dtype):
    # nan/inf/-inf/+-0.0 are ordinary stored values: _values must return them
    # verbatim (equal_nan=True), never sanitized.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


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
    # SparseCsr* backends have no kernel for _values: a CSR tensor raises
    # instead of returning its storage values, and the candidate must fail
    # too.
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
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
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        torch.ops.aten._values(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)
