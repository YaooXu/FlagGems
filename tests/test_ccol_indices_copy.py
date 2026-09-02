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

import math
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
from . import test_utils as tu  # noqa: E402

# aten::ccol_indices_copy(Tensor self) -> Tensor materializes the
# batch_dims + (n_cols + 1,) (CSC) / batch_dims + (n_col_blocks + 1,) (BSC)
# int64 compressed-column index tensor of a sparse column-compressed tensor as
# a fresh, contiguous, independent copy (the view_copy counterpart of
# aten::ccol_indices, whose native body is `ccol_indices(self).clone(
# contiguous)`). Every workload feeds a sparse CSC or BSC tensor and checks
# copy semantics: the result must equal the raw ccol array, must NOT alias the
# input's internal ccol storage, and must not mutate the input. The returned
# tensor never depends on the stored values, so every supported storage dtype
# (all float/int families plus bool) is exercised through the same layouts.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (layout, size, nnz, blocks) cases from the quick/all
#     --quicks — 2-D CSC (incl. single-row, single-column, square, full
#     and nnz == 0), batched CSC, 2-D BSC with varied block shapes, and
#     batched BSC, ranks 2-5;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     supported storage dtype is exercised with negative, positive, extreme
#     and degenerate value ranges (the returned ccol copy is identical for all
#     of them);
#   * edge cases: empty (nnz == 0, CSC and BSC), uncoalesced (duplicate row
#     entries inside a column), and nan/inf/-0.0 values (all ignored by the
#     accessor);
#   * negative: dense tensors, CSR tensors, COO tensors and a wrong-dtype
#     ``out`` tensor are rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a
# fresh int64 metadata tensor (nothing to broadcast against or differentiate).

# (layout, size, nnz, blocks) core cases: 2-D CSC (incl. single-row/single-
# column/square/nnz==0), batched CSC, 2-D BSC with varied block shapes, batched
# BSC, and an empty BSC.
_CCOLS_CORE = [
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
    ("bsc", (4, 6), 0, (2, 2)),
    ("bsc_batch", (2, 4, 6), 6, (2, 2)),
]

# Higher-rank / wider layouts for the "all" level (default, no --quick): multi-batch-
# dim CSC, a BSC whose column blocks do not divide the column count, and a
# batched BSC with a bigger block.
_CCOLS_ALL = [
    ("csc_batch", (7, 3, 12, 4, 5), 48, None),
    ("bsc", (10, 10), 12, (3, 4)),
    ("bsc_batch", (2, 8, 12), 12, (4, 4)),
]


def _ccol_cases():
    """(layout, size, nnz, blocks) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return [("csc_batch", (2, 19, 7), 20, None)]
    if tu.LEVEL == "all":
        return _CCOLS_CORE + _CCOLS_ALL


def _ccol_value_range_cases():
    """Representative all-sparse + batched layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [("csc", (5, 4), 6, None)]
    if tu.LEVEL == "all":
        return [
            ("csc", (5, 4), 6, None),
            ("csc_batch", (2, 6, 8), 12, None),
            ("bsc", (4, 6), 4, (2, 2)),
        ]
    return [("csc", (5, 4), 6, None), ("bsc", (4, 6), 4, (2, 2))]


# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse CSC/BSC runtime supports: every float, int, and
# bool family.
_CCOLS_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage uses float + int storage dtypes (bool ignores the range
# in tu.make_input and adds nothing beyond the copy-semantics tests above).
_VALUE_RANGE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# nan / +-inf stored values for the metadata-accessor test.
_NAN_INF_DTYPES = utils.ALL_FLOAT_DTYPES


def _random_ccol(n_compressed, nnz, gen):
    # A valid compressed-column array: length n_compressed + 1, non-decreasing,
    # ccol[0] == 0 and ccol[-1] == nnz. With n_compressed == 1 the array is the
    # degenerate [0, nnz].
    if n_compressed == 1:
        return torch.tensor([0, nnz], dtype=torch.long)
    inner = torch.sort(
        torch.randint(0, nnz + 1, (n_compressed - 1,), dtype=torch.long, generator=gen)
    ).values
    return torch.cat(
        [torch.zeros(1, dtype=torch.long), inner, torch.tensor([nnz], dtype=torch.long)]
    )


def _make_values(dtype, values_shape, value_range, gen):
    # Stored values come from the value-range framework (tu.make_input) because
    # the returned ccol copy never depends on them; every per-dtype range can
    # therefore be exercised through this one constructor. bool ignores the
    # range (deterministic random 0/1 keeps the seed-based construction
    # reproducible).
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen).to(
            flag_gems.device
        )
    # tu.make_input already creates the tensor on flag_gems.device; the .to()
    # it to flag_gems.device so it always matches the index tensors.
    return tu.make_input(dtype, values_shape, list(value_range)).to(flag_gems.device)


def _make_csc(size, nnz, dtype, gen, device, value_range):
    n_rows, n_cols = size
    ccol = _random_ccol(n_cols, nnz, gen)
    rows = torch.randint(0, n_rows, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz,), value_range, gen)
    return torch.sparse_csc_tensor(ccol, rows, values, size=size, device=device)


def _make_csc_batch(size, nnz, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    n_batch = 1
    for dim in batch_dims:
        n_batch *= dim
    ccol = torch.stack([_random_ccol(n_cols, nnz, gen) for _ in range(n_batch)])
    rows = torch.randint(0, n_rows, (n_batch, nnz), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (n_batch, nnz), value_range, gen)
    return torch.sparse_csc_tensor(
        ccol.view(batch_dims + (n_cols + 1,)),
        rows.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz,)),
        size=size,
        device=device,
    )


def _make_bsc(size, nnz, blocks, dtype, gen, device, value_range):
    n_rows, n_cols = size
    block_rows, block_cols = blocks
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    ccol = _random_ccol(n_col_blocks, nnz, gen)
    row = torch.randint(0, n_row_blocks, (nnz,), dtype=torch.long, generator=gen)
    values = _make_values(dtype, (nnz, block_rows, block_cols), value_range, gen)
    # torch.sparse_bsc_tensor infers the block size from the trailing dims of
    # the values tensor (values_shape == (nnz, block_rows, block_cols)), so
    # blocks is only used to size the index arrays here.
    return torch.sparse_bsc_tensor(ccol, row, values, size=size, device=device)


def _make_bsc_batch(size, nnz, blocks, dtype, gen, device, value_range):
    batch_dims, n_rows, n_cols = size[:-2], size[-2], size[-1]
    block_rows, block_cols = blocks
    n_batch = 1
    for dim in batch_dims:
        n_batch *= dim
    n_col_blocks = int(math.ceil(n_cols / block_cols))
    n_row_blocks = int(math.ceil(n_rows / block_rows))
    ccol = torch.stack([_random_ccol(n_col_blocks, nnz, gen) for _ in range(n_batch)])
    row = torch.randint(
        0, n_row_blocks, (n_batch, nnz), dtype=torch.long, generator=gen
    )
    values = _make_values(
        dtype, (n_batch, nnz, block_rows, block_cols), value_range, gen
    )
    return torch.sparse_bsc_tensor(
        ccol.view(batch_dims + (n_col_blocks + 1,)),
        row.view(batch_dims + (nnz,)),
        values.view(batch_dims + (nnz, block_rows, block_cols)),
        size=size,
        device=device,
    )


def _make_input(layout, size, nnz, blocks, dtype, value_range=("-1", "1"), seed=0):
    # Deterministic CPU-side compressed/row index generation; the values tensor
    # comes from the shared value-range helper (tu.make_input) and the sparse
    # tensor is created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    if layout == "csc":
        return _make_csc(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "csc_batch":
        return _make_csc_batch(size, nnz, dtype, gen, flag_gems.device, value_range)
    if layout == "bsc":
        return _make_bsc(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)
    return _make_bsc_batch(size, nnz, blocks, dtype, gen, flag_gems.device, value_range)


def _expected_ccol_shape(case):
    layout, size, nnz, blocks = case
    del nnz
    if layout == "csc":
        return (size[-1] + 1,)
    if layout == "csc_batch":
        return size[:-2] + (size[-1] + 1,)
    n_col_blocks = int(math.ceil(size[-1] / blocks[1]))
    if layout == "bsc":
        return (n_col_blocks + 1,)
    return size[:-2] + (n_col_blocks + 1,)


def _reference_ccol_indices_copy(inp):
    # Prefer the literal ATen op as the reference. The installed PyTorch build
    # registers ccol_indices_copy as CompositeExplicitAutogradNonFunctional,
    # whose dispatch-key set may exclude the SparseCsr functionality key, so
    # calling torch.ops.aten.ccol_indices_copy directly on a sparse tensor can
    # raise NotImplementedError. In that case fall back to the operator's exact
    # native body -- ccol_indices(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse tensors.
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
    # Same strategy as _reference_ccol_indices_copy for the .out overload. The
    # .out contract materializes int64 entries into out and returns out itself;
    # ATen enforces the int64 out dtype, so the manual native-body fallback
    # (compute the materialized copy and copy it into out) enforces the same
    # contract up front to stay consistent across builds where the .out op is
    # unreachable on sparse tensors.
    if out.dtype != torch.int64:
        raise RuntimeError(
            "Expected out tensor to have dtype long int, "
            f"but got {out.dtype} instead"
        )
    try:
        return torch.ops.aten.ccol_indices_copy.out(inp, out=out)
    except NotImplementedError:
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


def _assert_copy_semantics(res, ref, inp, ref_inp, expected_shape):
    # ccol_indices_copy returns a fresh contiguous int64 tensor holding the
    # input's raw compressed-column array. The result must not alias the
    # input's internal ccol storage and the input must not be mutated.
    assert res.dtype == torch.int64
    assert ref.dtype == torch.int64
    assert res.shape == expected_shape
    assert ref.shape == expected_shape
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's ccol array.
    assert res.data_ptr() != inp.ccol_indices().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set). equal_nan=True keeps the
    # non-mutation check valid for inputs whose stored values contain nan /
    # +-inf (test_ccol_indices_copy_nan_inf_values): a mutated tensor still
    # differs from the snapshot on the finite entries.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _ccol_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _ccol_cases())
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out(case, dtype):
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(_expected_ccol_shape(case), dtype=torch.long, device=inp.device)
    ref_out = torch.empty(
        _expected_ccol_shape(case), dtype=torch.long, device=ref_inp.device
    )

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("case", _ccol_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_ccol_indices_copy_value_ranges(case, value_range, dtype):
    # Value-range coverage from the regular-operator spec: the metadata output
    # (the fresh ccol int64 copy) is independent of the stored values, so every
    # per-dtype range -- including the extreme [min, 0] and [0, max] magnitudes
    # -- must be accepted and must not perturb the returned ccol array.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("case", _ccol_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test_ccol_indices_copy_out_value_ranges(case, value_range, dtype):
    # Same sweep through the .out overload: the int64 ccol copy written into
    # out must be identical for every per-dtype value range of the storage.
    layout, size, nnz, blocks = case
    inp = _make_input(layout, size, nnz, blocks, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(_expected_ccol_shape(case), dtype=torch.long, device=inp.device)
    ref_out = torch.empty(
        _expected_ccol_shape(case), dtype=torch.long, device=ref_inp.device
    )

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, _expected_ccol_shape(case))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_empty_bsc(dtype):
    # nnz == 0 for BSC: rows and values are empty, but ccol_indices_copy must
    # still return a (n_col_blocks + 1,) contiguous int64 tensor (not a dense
    # or wrongly-shaped tensor).
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (4,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out_empty_bsc(dtype):
    inp = _make_input("bsc", (4, 6), 0, (2, 2), dtype)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(4, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(4, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (4,))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (rows[0] == rows[1] in column 0), which
    # leaves the CSC tensor with repeated entries; ccol_indices_copy must still
    # return exactly the stored ccol array, in storage order, as an independent
    # copy (never a coalesced/sorted array and never an alias). Column 0 holds
    # 3 entries for rows [0, 0, 2], so a coalescing implementation would
    # visibly change the stored structure.
    shape = (3, 4)
    ccol = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _CCOLS_DTYPES)
def test_ccol_indices_copy_out_uncoalesced(dtype):
    shape = (3, 4)
    ccol = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (5,), ["-1", "1"], torch.Generator("cpu"))
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_ccol_indices_copy_nan_inf_values(dtype):
    # nan / +-inf stored values must not perturb the returned ccol copy:
    # ccol_indices_copy reads only the compressed-index storage, so the copy
    # must still be bit-exact even when the values contain non-finite entries.
    shape = (3, 4)
    ccol = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_ccol_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy_out
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test_ccol_indices_copy_out_nan_inf_values(dtype):
    shape = (3, 4)
    ccol = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = torch.sparse_csc_tensor(ccol, rows, values, shape)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.long, device=ref_inp.device)

    ref_ret = _reference_ccol_indices_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp, (5,))


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_dense():
    # ccol_indices_copy is a column-compressed-sparse-only metadata accessor: a
    # dense tensor has no compressed column index storage, so both the
    # reference and the candidate must reject it.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_csr():
    # SparseCsr is a distinct compressed layout from SparseCsc: a CSR tensor
    # stores crow/col pointers instead of a compressed column array and must be
    # rejected.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy
def test_ccol_indices_copy_negative_coo():
    # Sparse (COO) is a distinct backend key from SparseCsr (CSC); ccol_indices
    # has no Sparse implementation and must reject COO tensors. The reference
    # primary raises NotImplementedError, and the native-body fallback
    # (ccol_indices) raises RuntimeError, so both are covered here.
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_ccol_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_dense():
    # The .out variant is equally column-compressed-sparse-only: a dense tensor
    # has no compressed column index storage and must be rejected.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_csr():
    # The .out variant is equally column-compressed-sparse-only: a CSR tensor
    # is rejected.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_coo():
    # The .out variant is equally column-compressed-sparse-only: a COO tensor
    # is rejected (the reference primary raises NotImplementedError, and the
    # native-body fallback raises RuntimeError).
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.randn(2, dtype=torch.float32)
    inp = torch.sparse_coo_tensor(indices, values, (3, 3), device=flag_gems.device)
    out = torch.empty(5, dtype=torch.long, device=inp.device)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _reference_ccol_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark.ccol_indices_copy_out
def test_ccol_indices_copy_out_negative_wrong_dtype():
    # The .out contract materializes int64 entries into out; an out tensor of a
    # different dtype must be rejected (ATen raises RuntimeError, and the
    # manual fallback enforces the same contract up front).
    inp = _make_input("csc", (5, 4), 6, None, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty(5, dtype=torch.float32, device=inp.device)
    ref_out = torch.empty(5, dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_ccol_indices_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
