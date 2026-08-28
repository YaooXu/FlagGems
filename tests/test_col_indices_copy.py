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

# aten::col_indices_copy(Tensor self) -> Tensor materializes the column index
# array of a sparse row-compressed tensor (CSR or BSR) as a fresh, contiguous,
# independent copy (the view_copy counterpart of aten::col_indices, whose
# native body is `col_indices(self).clone(contiguous)`). Every workload feeds a
# row-compressed sparse tensor and checks copy semantics: the result must equal
# the raw col_indices, must NOT alias the input's internal column-index
# storage, and must not mutate the input. Each (layout, size, nnz, blocks)
# tuple is a distinct layout: 2-D CSR (including single-row / single-column and
# the nnz == 0 boundary), batched CSR with regular nnz per batch, 2-D BSR with
# block sizes, and batched BSR.
_COLS = [
    ("csr", (5, 4), 6, None),
    ("csr", (4, 1), 3, None),
    ("csr", (1, 5), 2, None),
    ("csr", (8, 8), 16, None),
    ("csr", (16, 32), 40, None),
    ("csr", (32, 16), 80, None),
    ("csr", (3, 3), 9, None),
    ("csr", (3, 4), 0, None),
    ("csr_batch", (2, 6, 8), 12, None),
    ("bsr", (4, 6), 4, (2, 2)),
    ("bsr", (8, 8), 8, (2, 2)),
    ("bsr", (6, 6), 6, (3, 2)),
    ("bsr_batch", (2, 4, 6), 6, (2, 2)),
]

# The result is always int64 and ignores the stored values, but the candidate
# must accept every storage dtype the sparse row-compressed runtime supports:
# every float, int, and bool family.
_COLS_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _random_values(shape, dtype, gen):
    # Deterministic CPU-side generation for every sparse storage dtype.
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    # Keep the magnitude small so the values stay valid for every integer
    # storage dtype.
    return torch.randint(-5, 6, shape, dtype=dtype, generator=gen)


def _random_crow(n_compressed, nnz, gen):
    # Non-decreasing compressed row index array of length n_compressed + 1 with
    # crow[0] == 0 and crow[-1] == nnz. Repeated split points leave empty rows,
    # which is valid for the compressed format.
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
    if layout == "csr":
        return _make_csr(size, nnz, dtype, gen, device)
    if layout == "csr_batch":
        return _make_csr_batch(size, nnz, dtype, gen, device)
    if layout == "bsr":
        return _make_bsr(size, nnz, blocks, dtype, gen, device)
    if layout == "bsr_batch":
        return _make_bsr_batch(size, nnz, blocks, dtype, gen, device)
    raise ValueError(f"unknown layout {layout}")


def _make_csr(size, nnz, dtype, gen, device):
    n_rows, n_cols = size
    crow = _random_crow(n_rows, nnz, gen)
    col = torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen)
    values = _random_values((nnz,), dtype, gen)
    return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)


def _make_csr_batch(size, nnz, dtype, gen, device):
    batch, n_rows, n_cols = size
    crows, cols, values = [], [], []
    for _ in range(batch):
        crows.append(_random_crow(n_rows, nnz, gen))
        cols.append(torch.randint(0, n_cols, (nnz,), dtype=torch.long, generator=gen))
        values.append(_random_values((nnz,), dtype, gen))
    return torch.sparse_csr_tensor(
        torch.stack(crows),
        torch.stack(cols),
        torch.stack(values),
        size=size,
        device=device,
    )


def _make_bsr(size, nnz, blocks, dtype, gen, device):
    n_rows, n_cols = size
    br, bc = blocks
    n_row_blocks = n_rows // br
    n_col_blocks = n_cols // bc
    crow = _random_crow(n_row_blocks, nnz, gen)
    col = torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
    values = _random_values((nnz, br, bc), dtype, gen)
    return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)


def _make_bsr_batch(size, nnz, blocks, dtype, gen, device):
    batch, n_rows, n_cols = size
    br, bc = blocks
    n_row_blocks = n_rows // br
    n_col_blocks = n_cols // bc
    crows, cols, values = [], [], []
    for _ in range(batch):
        crows.append(_random_crow(n_row_blocks, nnz, gen))
        cols.append(
            torch.randint(0, n_col_blocks, (nnz,), dtype=torch.long, generator=gen)
        )
        values.append(_random_values((nnz, br, bc), dtype, gen))
    return torch.sparse_bsr_tensor(
        torch.stack(crows),
        torch.stack(cols),
        torch.stack(values),
        size=size,
        device=device,
    )


def _reference_col_indices_copy(inp):
    # Prefer the literal ATen op as the reference. Some PyTorch builds register
    # col_indices_copy as CompositeExplicitAutogradNonFunctional whose
    # dispatch-key set excludes the SparseCsr functionality key, in which case
    # calling torch.ops.aten.col_indices_copy directly on a sparse tensor
    # raises NotImplementedError. In that case fall back to the operator's
    # exact native body -- col_indices(self).clone(contiguous) -- composed
    # from ATen ops, which IS reachable on sparse row-compressed tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten.col_indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten.col_indices(inp).clone(
            memory_format=torch.contiguous_format
        )


def _reference_col_indices_copy_out(inp, out):
    # Same strategy as _reference_col_indices_copy for the .out overload:
    # compute the materialized copy and write it into out (the .out contract
    # returns out itself).
    computed = _reference_col_indices_copy(inp)
    torch.ops.aten.copy_(out, computed)
    return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.col_indices_copy is registered;
    # resolution order is: (1) override, (2) the direct
    # flag_gems.col_indices_copy callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "col_indices_copy", getattr(flag_gems, "col_indices_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "col_indices_copy.out", getattr(flag_gems, "col_indices_copy_out", None)
    )


def _expected_col_shape(case):
    layout, size, nnz, blocks = case
    del blocks
    if layout == "csr":
        return (nnz,)
    if layout == "csr_batch":
        return (size[0], nnz)
    if layout == "bsr":
        return (nnz,)
    if layout == "bsr_batch":
        return (size[0], nnz)
    raise ValueError(f"unknown layout {layout}")


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_shape):
    # col_indices_copy returns a fresh contiguous int64 tensor holding the
    # input's column index array (nnz entries, or batch x nnz for batched
    # layouts). The result must not alias the input's internal column-index
    # storage and the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == expected_shape
    assert ref.shape == expected_shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's col_indices.
    # For nnz == 0 the result and the input's internal col_indices storage are
    # both empty (data_ptr() == 0), so the pointer check only applies to
    # non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != inp.col_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set).
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.col_indices_copy
@pytest.mark.parametrize("case", _COLS)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_col_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_col_shape(case))


@pytest.mark.col_indices_copy_out
@pytest.mark.parametrize("case", _COLS)
@pytest.mark.parametrize("dtype", _COLS_DTYPES)
def test_col_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    expected_shape = _expected_col_shape(case)
    out = torch.empty(expected_shape, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(expected_shape, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_col_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, expected_shape)
