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

import sys as _sys
from pathlib import Path as _Path

# pytest --import-mode=importlib imports this module as <pkg>.test__coalesce,
# where <pkg> is the "tests" or "benchmark" package of the checkout that
# actually holds this file (the KernelGen verification harness stages a temp
# copy of the FlagGems tree). When the driving process also has a same-named
# package on sys.path (e.g. the KernelGen repo's own tests/ directory), a bare
# relative import below would bind to that foreign package instead. Put the
# checkout root of *this* file first in sys.path so the relative imports
# resolve to the support files (accuracy_utils/test_utils/base/consts) that
# ship next to it.
_CHECKOUT_ROOT = _Path(__file__).resolve().parent.parent
if str(_CHECKOUT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CHECKOUT_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402
from . import test_utils as tu  # noqa: E402

# ``_coalesce`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._coalesce`` and ``-m
# _coalesce`` both work.
setattr(
    pytest.mark,
    "_coalesce",
    MarkDecorator(Mark("_coalesce", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_coalesce_out",
    MarkDecorator(Mark("_coalesce_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_coalesce(Tensor self) -> Tensor merges duplicate entries of an
# uncoalesced sparse COO tensor: the result has unique, lexicographically
# sorted indices and values equal to the sum of the entries sharing each
# index. Every case below picks nnz > numel so the pigeonhole principle
# guarantees duplicate indices and coalescing always has work to do. The CUDA
# reference asserts on already-coalesced inputs, so all inputs stay
# uncoalesced and coalescing is never allowed to mutate them in place.
#
# Shape levels: 1-D through 4-D sparse COO tensors. Broadcast does not apply
# to this unary sparse op, and sparse COO autograd has no formula for
# ``_coalesce``, so there is no broadcast/backward test.
_COALESCE_CASES = [
    ((4, 4), 20),
    ((5, 5), 30),
    ((8, 8), 80),
    ((16, 16), 300),
    ((64,), 200),
    ((2, 3, 4), 28),
    ((3, 5, 7), 120),
    ((4, 8, 16), 600),
    ((4, 4, 4, 4), 400),
]

# Summing duplicates is exact for every integer/bool storage dtype aten
# supports; float summation order may differ between implementations, so float
# results are compared with the dtype-appropriate tolerance and integer/bool
# results with exact equality.
_COALESCE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage (regular-operator spec). Coalescing *sums* duplicate
# values, so the shared ``tu.selected_ranges()`` set is not usable verbatim:
# ``[0, max]`` / ``[min, 0]`` would overflow the smallest float dtypes when
# several duplicates accumulate, and ``[-1, 1]`` can cancel to a near-zero sum
# whose fp16/bf16 summation-order error exceeds the tolerance. Same-sign
# ranges are therefore used for every float dtype (each dtype gets both a
# negative and a positive range, resolved per-dtype by ``tu.resolve_bound``)
# and the mixed-sign ``[-1, 1]`` range is restricted to fp32/fp64, where the
# cancellation error stays far below the tolerance.
_VALUE_RANGE_CASES = [
    (range_symbols, dtype)
    for dtype in utils.ALL_FLOAT_DTYPES
    for range_symbols in [("0", "1"), ("-1", "0")]
] + [
    (("-1", "1"), dtype)
    for dtype in utils.ALL_FLOAT_DTYPES
    if dtype in (torch.float32, torch.float64)
]


def _make_input(shape, nnz, dtype, low=None, high=None):
    # Deterministic CPU-side generation (then the sparse tensor is created on
    # the test device). Index rows are drawn with replacement, so duplicates
    # are guaranteed whenever nnz > numel.
    gen = torch.Generator("cpu").manual_seed(2026)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    if dtype.is_floating_point:
        # Non-negative values by default: coalescing sums duplicates, and
        # summing in different orders can differ by ulps in fp16/bf16. With
        # same-sign values there is no cancellation, so any summation order
        # stays within the dtype tolerance (this keeps the CPU reference and
        # the device candidate comparable in --ref cpu mode). ``low``/``high``
        # let the value-range test draw signed values from resolved bounds.
        if low is None:
            low, high = 0.0, 1.0
        values = (
            torch.rand(nnz, dtype=torch.float32, generator=gen) * (high - low) + low
        ).to(dtype)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    else:
        # Keep the magnitude small so summed duplicates cannot overflow the
        # smallest integer dtype (int16).
        values = torch.randint(-5, 6, (nnz,), dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_nan_inf_values(nnz, dtype):
    # Repeating pattern of nan / +inf / -inf / finite values. Coalescing maps
    # an index to nan when any contributor is nan or both +inf and -inf
    # contribute; the nan/inf position pattern is fully determined by the
    # value multiset, so the CPU reference and the device candidate agree
    # exactly (verified with equal_nan=True).
    base = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.5,
            -0.5,
            2.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            1.5,
        ],
        dtype=dtype,
    )
    return base.repeat((nnz + base.numel() - 1) // base.numel())[:nnz]


def _make_empty_out(shape, dtype, device):
    # Empty sparse COO tensor of the right shape/dtype; _coalesce.out writes
    # the coalesced indices and values into this storage.
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_coalesce" and "_coalesce.out".
    return flag_gems.testing.resolve_gems_op(
        "_coalesce", getattr(flag_gems, "_coalesce", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_coalesce.out", getattr(flag_gems, "_coalesce_out", None)
    )


def _assert_coalesced(res_out, ref_out, dtype):
    # Both sides must be coalesced sparse COO tensors with the same structure.
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced()
    assert ref_out.is_coalesced()
    # Indices are int64 and must match exactly (unique, sorted index pairs).
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    # Values are summed duplicates; use the dtype-appropriate tolerance.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # Coalescing returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce_out
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce_out(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = _make_empty_out(shape, dtype, flag_gems.device)
    ref_out = _make_empty_out(shape, dtype, ref_inp.device)

    ref_ret = torch.ops.aten._coalesce.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_coalesced(res_ret, ref_ret, dtype)
    assert not inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize(("range_symbols", "dtype"), _VALUE_RANGE_CASES)
def test__coalesce_value_ranges(case, range_symbols, dtype):
    shape, nnz = case
    low = tu.resolve_bound(range_symbols[0], dtype)
    high = tu.resolve_bound(range_symbols[1], dtype)
    inp = _make_input(shape, nnz, dtype, low=float(low), high=float(high))
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__coalesce_nan_inf(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    # Rebuild with the same duplicate indices but values drawn from the
    # nan/inf/-inf pattern (the nan/inf result pattern is deterministic).
    inp = torch.sparse_coo_tensor(
        inp._indices().clone(),
        _make_nan_inf_values(nnz, dtype),
        shape,
        device=flag_gems.device,
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    # Both sides must be coalesced; .indices() on an uncoalesced tensor
    # raises, so this also proves the structure is right.
    assert res_out.is_coalesced()
    assert ref_out.is_coalesced()
    # nan == nan under equal_nan=True; inf signs must match exactly.
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype, equal_nan=True)
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce
def test__coalesce_rejects_dense_input():
    # _coalesce is a sparse-COO-only operator; a dense (strided) input has no
    # registered kernel and must raise. NotImplementedError is a RuntimeError
    # subclass, so the candidate is held to the same contract on any device.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)


@pytest.mark._coalesce
def test__coalesce_rejects_csr_input():
    # Sparse CSR layout is not COO: _coalesce must reject it as well.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
