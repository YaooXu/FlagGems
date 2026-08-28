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

import flag_gems

# KernelGen's verification harness stages the test files in a temporary
# copy of the FlagGems tree and runs pytest with --import-mode=importlib
# from a working directory that is not on sys.path, so the parent of this
# package may not be importable yet.  Make it importable before using the
# relative import below.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from . import accuracy_utils as utils  # noqa: E402

# aten::ccol_indices_copy(Tensor self) -> Tensor materializes the compressed
# column index array of a sparse column-compressed tensor (CSC or BSC) as a
# fresh, contiguous, independent copy (the view_copy counterpart of
# aten::ccol_indices, whose native body is `ccol_indices(self).clone(
# contiguous)`). Every workload feeds a column-compressed sparse tensor and
# checks copy semantics: the result must equal the raw ccol_indices, must NOT
# alias the input's internal compressed-index storage, and must not mutate the
# input. Each (layout, size, nnz, blocks) tuple is a distinct layout: 2-D CSC
# (including single-row / single-column and the nnz == 0 boundary), batched
# CSC with regular nnz per batch, 2-D BSC with block sizes, and batched BSC.
_CCOLS = [
    ("csc", (5, 4), 6, None),
    ("csc", (4, 1), 3, None),
    ("csc", (1, 5), 2, None),
    ("csc", (8, 8), 16, None),
    ("csc", (16, 32), 40, None),
    ("csc", (32, 16), 80, None),
    ("csc", (3, 3), 9, None),
    ("csc", (3, 4), 0, None),
    ("csc_batch", (2, 6, 8), 12, None),
    ("bsc", (4, 6), 4, (2, 2)),
    ("bsc", (8, 8), 8, (2, 2)),
    ("bsc", (6, 6), 6, (3, 2)),
    ("bsc_batch", (2, 4, 6), 6, (2, 2)),
]

# The result is always int64 and ignores the stored values, but the candidate
# must accept every storage dtype the sparse column-compressed runtime
# supports: every float, int, and bool family.
_CCOLS_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _random_values(shape, dtype, gen):
    # Deterministic CPU-side generation for every sparse storage dtype.
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    # Keep the magnitude small so the values stay valid for every integer
    # storage dtype.
    return torch.randint(-5, 6, shape, dtype=dtype, generator=gen)


def _random_ccol(n_compressed, nnz, gen):
    # Non-decreasing compressed index array of length n_compressed + 1 with
    # ccol[0] == 0 and ccol[-1] == nnz. Repeated split points leave empty
    # columns, which is valid for the compressed format.
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
    )


def _make_input(layout, size, nnz, blocks, dtype, seed=0, device=flag_gems.device):
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csc":
        return _make_csc(size, nnz, dtype, gen, device)
    if layout == "csc_batch":
        return _make_csc_batch(size, nnz, dtype, gen, device)
    if layout == "bsc":
        return _make_bsc(size, nnz, blocks, dtype, gen, device)
    if layout == "bsc_batch":
        return _make_bsc_batch(size, nnz, blocks, dtype, gen, device)
    raise ValueError(f"unknown layout {layout}")


def _make_csc(size, nnz, dtype, gen, device):
    n_rows, n_cols = size
    ccol = _random_ccol(n_cols, nnz, gen)
    row = torch.randint(0, n_rows, (nnz,), dtype=torch.long, generator=gen)
    values = _random_values((nnz,), dtype, gen)
    return torch.sparse_csc_tensor(ccol, row, values, size=size, device=device)


def _make_csc_batch(size, nnz, dtype, gen, device):
    batch, n_rows, n_cols = size
    ccols, rows, values = [], [], []
    for _ in range(batch):
        ccols.append(_random_ccol(n_cols, nnz, gen))
        rows.append(torch.randint(0, n_rows, (nnz,), dtype=torch.long, generator=gen))
        values.append(_random_values((nnz,), dtype, gen))
    return torch.sparse_csc_tensor(
        torch.stack(ccols),
        torch.stack(rows),
        torch.stack(values),
        size=size,
        device=device,
    )


def _make_bsc(size, nnz, blocks, dtype, gen, device):
    n_rows, n_cols = size
    br, bc = blocks
    n_col_blocks = n_cols // bc
    ccol = _random_ccol(n_col_blocks, nnz, gen)
    row = torch.randint(0, n_rows // br, (nnz,), dtype=torch.long, generator=gen)
    values = _random_values((nnz, br, bc), dtype, gen)
    return torch.sparse_bsc_tensor(ccol, row, values, size=size, device=device)


def _make_bsc_batch(size, nnz, blocks, dtype, gen, device):
    batch, n_rows, n_cols = size
    br, bc = blocks
    n_col_blocks = n_cols // bc
    ccols, rows, values = [], [], []
    for _ in range(batch):
        ccols.append(_random_ccol(n_col_blocks, nnz, gen))
        rows.append(
            torch.randint(0, n_rows // br, (nnz,), dtype=torch.long, generator=gen)
        )
        values.append(_random_values((nnz, br, bc), dtype, gen))
    return torch.sparse_bsc_tensor(
        torch.stack(ccols),
        torch.stack(rows),
        torch.stack(values),
        size=size,
        device=device,
    )


def _reference_ccol_indices_copy(inp):
    # Prefer the literal ATen op as the reference. Some PyTorch builds register
    # ccol_indices_copy as CompositeExplicitAutogradNonFunctional whose
    # dispatch-key set excludes the SparseCsr functionality key, in which case
    # calling torch.ops.aten.ccol_indices_copy directly on a sparse tensor
    # raises NotImplementedError. In that case fall back to the operator's
    # exact native body -- ccol_indices(self).clone(contiguous) -- composed
    # from ATen ops, which IS reachable on sparse column-compressed tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten.ccol_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.ccol_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _reference_ccol_indices_copy_out(inp, out):
    # Same strategy as _reference_ccol_indices_copy for the .out overload:
    # compute the materialized copy and write it into out (the .out contract
    # returns out itself).
    computed = _reference_ccol_indices_copy(inp)
    torch.ops.aten.copy_(out, computed)
    return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.ccol_indices_copy is registered;
    # resolution order is: (1) override, (2) the direct
    # flag_gems.ccol_indices_copy callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices_copy", getattr(flag_gems, "ccol_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices_copy.out", getattr(flag_gems, "ccol_indices_copy_out", None)
    )


def _expected_ccol_shape(case):
    layout, size, nnz, blocks = case
    del nnz
    if layout == "csc":
        return (size[1] + 1,)
    if layout == "csc_batch":
        return (size[0], size[2] + 1)
    if layout == "bsc":
        return (size[1] // blocks[1] + 1,)
    if layout == "bsc_batch":
        return (size[0], size[2] // blocks[1] + 1)
    raise ValueError(f"unknown layout {layout}")


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_shape):
    # ccol_indices_copy returns a fresh contiguous (compressed_extent + 1)
    # int64 tensor holding the input's compressed column index array. The
    # result must not alias the input's internal compressed-index storage and
    # the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == expected_shape
    assert ref.shape == expected_shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's ccol_indices.
    # A valid compressed tensor always has a non-empty ccol_indices array
    # (length >= 2), so the pointer check is always meaningful.
    assert res.data_ptr() != inp.ccol_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set).
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _CCOLS)
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _CCOLS)
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    expected_shape = _expected_ccol_shape(case)
    out = torch.empty(expected_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(expected_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, expected_shape)
