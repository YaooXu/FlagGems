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

import itertools
import os
import sys

import pytest
import torch

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

# aten::crow_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (nrows + 1,) int64 compressed row index tensor of a sparse CSR
# tensor as a fresh, contiguous, independent copy (the view_copy counterpart of
# aten::crow_indices, whose native body is
# `crow_indices(self).clone(contiguous)`). Every workload feeds a sparse CSR
# tensor and checks copy semantics: the result must equal the raw crow indices,
# must NOT alias the input's internal crow storage, and must not mutate the
# input. Each (shape, nnz) pair is a distinct layout: 2-D all-sparse, 3-D
# batched, and 4-D multi-batch-dims, with varying nnz so the shape of the result
# is exercised.
_CROW_COPY_CASES = [
    ((5, 4), 7),
    ((3, 8), 16),
    ((8, 3), 12),
    ((4, 4), 16),
    ((1, 6), 4),
    ((3, 5, 4), 7),
    ((2, 4, 6), 12),
    ((2, 3, 4, 5), 8),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSR runtime supports: every float, int, and bool
# family.
_CROW_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype, seed=0):
    # Deterministic CPU-side generation, then the sparse CSR tensor is created
    # on the test device. (row, col) pairs are drawn with replacement:
    # duplicate entries are allowed and merely leave the tensor uncoalesced
    # (covered explicitly below). The crow pointer array is built with a
    # row-wise bincount, so it is always a valid CSR structure.
    gen = torch.Generator("cpu").manual_seed(seed)
    nrows, ncols = shape[-2], shape[-1]
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, generator=gen)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, generator=gen)
    order = torch.argsort(rows * ncols + cols, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    counts = torch.stack(
        [
            torch.bincount(rows[idx], minlength=nrows)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (nrows,))
    crow = torch.zeros(batch + (nrows + 1,), dtype=torch.long)
    crow[..., 1:] = torch.cumsum(counts, -1)
    if dtype.is_floating_point:
        values = torch.randn(entries_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, entries_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, entries_shape, dtype=dtype, generator=gen)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _reference_crow_indices_copy(inp):
    # Prefer the literal ATen op as the reference. Some PyTorch builds register
    # crow_indices_copy as CompositeExplicitAutogradNonFunctional, whose
    # dispatch-key set excludes the SparseCsr functionality key, so calling
    # torch.ops.aten.crow_indices_copy directly on a sparse CSR tensor raises
    # NotImplementedError. In that case fall back to the operator's exact
    # native body -- crow_indices(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse CSR tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten.crow_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.crow_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _reference_crow_indices_copy_out(inp, out):
    # Same strategy as _reference_crow_indices_copy for the .out overload:
    # compute the materialized copy and write it into out (the .out contract
    # returns out itself).
    try:
        return torch.ops.aten.crow_indices_copy.out(inp, out=out)
    except NotImplementedError:
        computed = torch.ops.aten.crow_indices(inp).clone(
            memory_format=torch.contiguous_format
        )
        torch.ops.aten.copy_(out, computed)
        return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.crow_indices_copy is registered;
    # resolution order is: (1) override, (2) the direct flag_gems.crow_indices_copy
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "crow_indices_copy", getattr(flag_gems, "crow_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "crow_indices_copy.out", getattr(flag_gems, "crow_indices_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp):
    # crow_indices_copy returns a fresh contiguous batch_dims + (nrows + 1,)
    # int64 tensor holding the input's raw crow indices. The entries are exact,
    # and copy semantics require the result to NOT alias the input's crow
    # storage (unlike aten::crow_indices) and to not mutate the input.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert ref.shape == inp.shape[:-2] + (inp.shape[-2] + 1,)
    assert res.shape == ref.shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's crow indices.
    # Zero-size results (nrows + 1 == 0) may share the null pointer, so only
    # assert on non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != torch.ops.aten.crow_indices(inp).data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set).
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("case", _CROW_COPY_CASES)
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("case", _CROW_COPY_CASES)
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy_out(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out_shape = shape[:-2] + (shape[-2] + 1,)
    out = torch.empty(out_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(out_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_crow_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy_empty(dtype):
    # nnz == 0: cols and values are empty, but crow_indices_copy must still
    # return a (nrows + 1,) contiguous int64 tensor (not a dense or
    # wrongly-shaped tensor).
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy_out
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy_out_empty(dtype):
    # nnz == 0: the .out variant must still write a (nrows + 1,) int64 tensor
    # into out and return out itself.
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out_shape = shape[:-2] + (shape[-2] + 1,)
    out = torch.empty(out_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(out_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_crow_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy_single_row(dtype):
    # nrows == 1: the returned crow has the degenerate shape (2,) with
    # crow[0] == 0 and crow[1] == nnz.
    shape, nnz = (1, 7), 5
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.crow_indices_copy
@pytest.mark.parametrize("dtype", _CROW_COPY_DTYPES)
def test_crow_indices_copy_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (cols[0] == cols[1] in row 0), which
    # leaves the tensor uncoalesced; crow_indices_copy must still return exactly
    # the stored crow tensor as an independent copy (never a coalesced/sorted
    # structure and never an alias). Row 0 holds 3 entries for columns
    # [0, 0, 2], so a coalescing implementation would visibly change the stored
    # structure.
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_crow_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)
