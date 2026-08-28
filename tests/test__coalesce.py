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
_COALESCE_CASES = [
    ((4, 4), 20),
    ((5, 5), 30),
    ((8, 8), 80),
    ((16, 16), 300),
    ((2, 3, 4), 28),
    ((3, 5, 7), 120),
    ((4, 8, 16), 600),
]

# Summing duplicates is exact for every integer/bool storage dtype aten
# supports; float summation order may differ between implementations, so float
# results are compared with the dtype-appropriate tolerance and integer/bool
# results with exact equality.
_COALESCE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype):
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
        # Non-negative values: coalescing sums duplicates, and summing in
        # different orders can differ by ulps in fp16/bf16. With same-sign
        # values there is no cancellation, so any summation order stays within
        # the dtype tolerance (this keeps the CPU reference and the device
        # candidate comparable in --ref cpu mode).
        values = torch.rand(nnz, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    else:
        # Keep the magnitude small so summed duplicates cannot overflow the
        # smallest integer dtype (int16).
        values = torch.randint(-5, 6, (nnz,), dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


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
