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

# aten::ccol_indices(Tensor(a) self) -> Tensor(a) returns the compressed column
# index tensor of a sparse CSC tensor: shape batch_dims + (ncols + 1,) (the
# unbatched 2-D case has batch_dims == ()) with dtype int64. The result is an
# alias of the input's internal ccol storage and is independent of the stored
# values, so every workload below feeds a sparse CSC tensor. Each (shape, nnz)
# pair is a distinct layout: 2-D all-sparse, 3-D batched, and 4-D
# multi-batch-dims, with varying nnz so the shape of the result is exercised.
_CCOL_CASES = [
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
# storage dtype the sparse CSC runtime supports: every float, int, and bool
# family.
_CCOL_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype, seed=0):
    # Deterministic CPU-side generation, then the sparse CSC tensor is created
    # on the test device. (row, col) pairs are drawn with replacement:
    # duplicate entries are allowed and merely leave the tensor uncoalesced
    # (covered explicitly below). The ccol pointer array is built with a
    # column-wise bincount, so it is always a valid CSC structure.
    gen = torch.Generator("cpu").manual_seed(seed)
    nrows, ncols = shape[-2], shape[-1]
    batch = shape[:-2]
    entries_shape = batch + (nnz,)
    rows = torch.randint(0, nrows, entries_shape, dtype=torch.long, generator=gen)
    cols = torch.randint(0, ncols, entries_shape, dtype=torch.long, generator=gen)
    order = torch.argsort(cols * nrows + rows, dim=-1)
    rows = torch.gather(rows, -1, order)
    cols = torch.gather(cols, -1, order)
    counts = torch.stack(
        [
            torch.bincount(cols[idx], minlength=ncols)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (ncols,))
    ccol = torch.zeros(batch + (ncols + 1,), dtype=torch.long)
    ccol[..., 1:] = torch.cumsum(counts, -1)
    if dtype.is_floating_point:
        values = torch.randn(entries_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, entries_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, entries_shape, dtype=dtype, generator=gen)
    return torch.sparse_csc_tensor(
        ccol.to(flag_gems.device),
        rows.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.ccol_indices is registered; resolution
    # order is: (1) override, (2) the direct flag_gems.ccol_indices callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "ccol_indices", getattr(flag_gems, "ccol_indices", None)
    )


def _assert_result(res_out, ref_out, inp, ref_inp):
    # ccol_indices returns a fresh view of the input's internal
    # batch_dims + (ncols + 1,) int64 compressed column index tensor. The
    # entries are exact, and the schema annotation Tensor(a) self -> Tensor(a)
    # requires the result to alias the input's ccol storage.
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert ref_out.shape == inp.shape[:-2] + (inp.shape[-1] + 1,)
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)
    # Alias semantics: the returned tensor shares storage with the input's
    # internal ccol tensor.
    assert res_out.data_ptr() == torch.ops.aten.ccol_indices(inp).data_ptr()
    assert ref_out.data_ptr() == torch.ops.aten.ccol_indices(ref_inp).data_ptr()
    # The accessor must not mutate the input: the result still matches the
    # (untouched) ccol captured on the reference copy before the call.
    utils.gems_assert_equal(res_out, torch.ops.aten.ccol_indices(ref_inp))


@pytest.mark.ccol_indices
@pytest.mark.parametrize("case", _CCOL_CASES)
@pytest.mark.parametrize("dtype", _CCOL_DTYPES)
def test_ccol_indices(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CCOL_DTYPES)
def test_ccol_indices_empty(dtype):
    # nnz == 0: rows and values are empty, but ccol_indices must still return a
    # (ncols + 1,) int64 tensor (not a dense or wrongly-shaped tensor).
    shape, nnz = (4, 5), 0
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CCOL_DTYPES)
def test_ccol_indices_single_column(dtype):
    # ncols == 1: the returned ccol has the degenerate shape (2,) with
    # ccol[0] == 0 and ccol[1] == nnz.
    shape, nnz = (7, 1), 5
    inp = _make_input(shape, nnz, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.ccol_indices
@pytest.mark.parametrize("dtype", _CCOL_DTYPES)
def test_ccol_indices_uncoalesced(dtype):
    # The (0, 0) entry is duplicated (rows[0] == rows[1] in column 0), which
    # leaves the tensor uncoalesced; ccol_indices must still return exactly the
    # stored ccol tensor (never a coalesced/sorted copy). Column 0 holds 3
    # entries for rows [0, 0, 2], so a coalescing implementation would visibly
    # change the stored structure.
    shape = (3, 4)
    ccol = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    rows = torch.tensor([0, 0, 2, 1, 3], dtype=torch.long, device=flag_gems.device)
    assert rows[0].item() == rows[1].item()
    gen = torch.Generator("cpu").manual_seed(0)
    if dtype.is_floating_point:
        values = torch.randn((5,), dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (5,), dtype=dtype, generator=gen)
    else:
        values = torch.randint(-5, 6, (5,), dtype=dtype, generator=gen)
    inp = torch.sparse_csc_tensor(ccol, rows, values.to(flag_gems.device), shape)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)
