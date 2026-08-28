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

# ``_indices_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._indices_copy`` and ``-m
# _indices_copy`` both work.
setattr(
    pytest.mark,
    "_indices_copy",
    MarkDecorator(Mark("_indices_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_indices_copy_out",
    MarkDecorator(Mark("_indices_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_indices_copy(Tensor self) -> Tensor materializes the (sparse_dim, nnz)
# int64 index tensor of a sparse COO tensor as a fresh, contiguous, independent
# copy (the view_copy counterpart of aten::_indices, whose native body is
# `_indices(self).clone(contiguous)`). Every workload feeds a sparse COO tensor
# and checks copy semantics: the result must equal the raw indices, must NOT
# alias the input's internal indices storage, and must not mutate the input.
# Each (shape, sparse_dim, nnz) triple is a distinct layout: 1-D all-sparse,
# 2-D/3-D all-sparse, and mixed sparse+dense ranks up to 5-D, with varying nnz
# so the (sparse_dim, nnz) shape of the result is exercised.
_INDICES_COPY_CASES = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse COO runtime supports: every float, int, and bool
# family.
_INDICES_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, sparse_dim, nnz, dtype, seed=0):
    # Deterministic CPU-side generation, then the sparse tensor is created on
    # the test device. Indices are drawn with replacement: duplicate indices
    # are allowed and merely leave the tensor uncoalesced (covered explicitly
    # below).
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    if dtype.is_floating_point:
        values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, (nnz,) + dense_shape, dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _reference_indices_copy(inp):
    # Prefer the literal ATen op as the reference. The installed PyTorch build
    # registers _indices_copy as CompositeExplicitAutogradNonFunctional, whose
    # dispatch-key set excludes the Sparse functionality key, so calling
    # torch.ops.aten._indices_copy directly on a sparse tensor raises
    # NotImplementedError. In that case fall back to the operator's exact
    # native body -- _indices(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    return torch.ops.aten._indices(inp).clone(memory_format=torch.contiguous_format)


def _reference_indices_copy_out(inp, out):
    # Same strategy as _reference_indices_copy for the .out overload: compute
    # the materialized copy and write it into out (the .out contract returns
    # out itself).
    computed = torch.ops.aten._indices(inp).clone(memory_format=torch.contiguous_format)
    torch.ops.aten.copy_(out, computed)
    return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._indices_copy is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._indices_copy callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_indices_copy", getattr(flag_gems, "_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_indices_copy.out", getattr(flag_gems, "_indices_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp):
    # _indices_copy returns a fresh contiguous (sparse_dim, nnz) int64 tensor
    # holding the input's raw indices. The result must not alias the input's
    # internal indices storage and the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == (inp.sparse_dim(), inp._nnz())
    assert ref.shape == (ref_inp.sparse_dim(), ref_inp._nnz())
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's indices.
    # Zero-size results (nnz == 0) may share the null pointer, so only assert
    # on non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != inp._indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set).
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark._indices_copy
@pytest.mark.parametrize("case", _INDICES_COPY_CASES)
@pytest.mark.parametrize("dtype", _INDICES_COPY_DTYPES)
def test__indices_copy(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy_out
@pytest.mark.parametrize("case", _INDICES_COPY_CASES)
@pytest.mark.parametrize("dtype", _INDICES_COPY_DTYPES)
def test__indices_copy_out(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(
        (inp.sparse_dim(), inp._nnz()), dtype=torch.long, device=inp.device
    )
    ref_out = torch.empty(
        (ref_inp.sparse_dim(), ref_inp._nnz()),
        dtype=torch.long,
        device=ref_inp.device,
    )

    ref_ret = _reference_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy
@pytest.mark.parametrize("dtype", _INDICES_COPY_DTYPES)
def test__indices_copy_empty(dtype):
    # nnz == 0: indices and values are empty, but _indices_copy must still
    # return a (sparse_dim, 0) contiguous int64 tensor (not a dense or
    # wrongly-shaped tensor).
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy_out
@pytest.mark.parametrize("dtype", _INDICES_COPY_DTYPES)
def test__indices_copy_out_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(
        (inp.sparse_dim(), inp._nnz()), dtype=torch.long, device=inp.device
    )
    ref_out = torch.empty(
        (ref_inp.sparse_dim(), ref_inp._nnz()),
        dtype=torch.long,
        device=ref_inp.device,
    )

    ref_ret = _reference_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy
@pytest.mark.parametrize("dtype", _INDICES_COPY_DTYPES)
def test__indices_copy_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _indices_copy must still
    # return exactly the stored index tensor, in storage order, as an
    # independent copy (never a coalesced/sorted tensor and never an alias).
    # The (0, 1) coordinate is repeated three times and the entries are NOT
    # sorted, so a coalescing implementation would visibly change the result.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)
