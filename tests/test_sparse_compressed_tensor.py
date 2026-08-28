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

# aten::sparse_compressed_tensor.comp_plain_value_size(Tensor compressed_indices,
#     Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# and the comp_plain_value (no-size) overload construct a sparse compressed
# tensor (CSR/CSC/BSR/BSC) from raw index tensors and values. The generic factory
# requires the layout kwarg and, unlike the torch.sparse_csr_tensor wrappers,
# does NOT infer the dtype from values: dtype= must be passed whenever the
# values dtype is not float32. On GPU the device= kwarg is required too (without
# it the instance is created on CPU and rejected by the same-device invariant in
# SparseCsrTensorImpl::set_member_tensors). Every workload below therefore calls
# both the reference and the candidate with dtype= and device= passed
# explicitly.
#
# Each (layout, shape, nnz, index_dtype) case is a distinct structure: 2-D
# CSR/CSC, 2-D block BSR/BSC (2x2 blocks), and 3-D/4-D batched CSR, with int64
# and int32 index tensors.
_SPARSE_COMPRESSED_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_csc, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
    (torch.sparse_bsc, (4, 6), 6, torch.int64),
    (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
    (torch.sparse_csr, (3, 5, 4), 7, torch.int32),
]

# The no-size overload estimates the shape from the indices instead of taking a
# size argument (compressed_dim = compressed.size(-1) - 1,
# plain_dim = max(plain) + 1, block sizes from the values shape).
_NO_SIZE_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_csc, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
]

# nnz == 0 for every layout: index tensors are empty but the compressed index
# tensor must still have the full compressed_dim + 1 entries.
_EMPTY_CASES = [
    (torch.sparse_csr, (4, 5)),
    (torch.sparse_csc, (4, 5)),
    (torch.sparse_bsr, (4, 4)),
    (torch.sparse_bsc, (4, 4)),
]

# The factory accepts every storage dtype the sparse compressed runtime
# supports: all float, all int, and bool.
_SPARSE_COMPRESSED_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2


def _make_input(layout, shape, nnz, dtype, index_dtype=torch.int64, seed=0):
    # Deterministic CPU-side generation; the index tensors and values are then
    # moved to the test device. (row, col) block entries are drawn with
    # replacement and sorted, and the compressed pointer array is built with a
    # row/column-wise bincount, so the structure is always a valid compressed
    # sparse tensor.
    device = flag_gems.device
    gen = torch.Generator("cpu").manual_seed(seed)
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    entries = batch + (nnz,)
    comp = torch.randint(0, comp_dim, entries, dtype=torch.long, generator=gen)
    plain = torch.randint(0, plain_dim, entries, dtype=torch.long, generator=gen)
    order = torch.argsort(comp * plain_dim + plain, dim=-1)
    comp = torch.gather(comp, -1, order)
    plain = torch.gather(plain, -1, order)
    counts = torch.stack(
        [
            torch.bincount(comp[idx], minlength=comp_dim)
            for idx in itertools.product(*(range(d) for d in batch))
        ]
    ).view(batch + (comp_dim,))
    compressed = torch.zeros(batch + (comp_dim + 1,), dtype=torch.long)
    compressed[..., 1:] = torch.cumsum(counts, -1)
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    if dtype.is_floating_point:
        values = torch.randn(entries + block_shape, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, entries + block_shape, dtype=dtype, generator=gen)
    else:
        # Keep the magnitude small so the values stay valid for every integer
        # storage dtype.
        values = torch.randint(-5, 6, entries + block_shape, dtype=dtype, generator=gen)
    return (
        compressed.to(device=device, dtype=index_dtype),
        plain.to(device=device, dtype=index_dtype),
        values.to(device),
    )


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems.sparse_compressed_tensor is
    # registered; resolution order is: (1) override, (2) the direct
    # flag_gems.sparse_compressed_tensor callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "sparse_compressed_tensor", getattr(flag_gems, "sparse_compressed_tensor", None)
    )


def _assert_result(res_out, ref_out, dtype, layout, index_dtype):
    # The constructed tensor must preserve the requested layout, dtype, shape,
    # and the exact index structure, and live on the test device.
    assert res_out.layout == layout
    assert ref_out.layout == layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert res_out.device.type == flag_gems.device
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)

    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_c = torch.ops.aten.crow_indices(res_out)
        ref_c = torch.ops.aten.crow_indices(ref_out)
    else:
        res_c = torch.ops.aten.ccol_indices(res_out)
        ref_c = torch.ops.aten.ccol_indices(ref_out)
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_p = torch.ops.aten.col_indices(res_out)
        ref_p = torch.ops.aten.col_indices(ref_out)
    else:
        res_p = torch.ops.aten.row_indices(res_out)
        ref_p = torch.ops.aten.row_indices(ref_out)
    assert res_c.dtype == index_dtype
    assert ref_c.dtype == index_dtype
    assert res_p.dtype == index_dtype
    assert ref_p.dtype == index_dtype
    # Indices are exact integer data.
    utils.gems_assert_equal(res_c, ref_c)
    utils.gems_assert_equal(res_p, ref_p)
    # Values follow the usual tolerance policy: floats are compared with
    # assert_close, integers and bools exactly.
    if dtype in utils.ALL_INT_DTYPES + utils.BOOL_TYPES:
        utils.gems_assert_equal(res_out.values(), ref_out.values())
    else:
        utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _SPARSE_COMPRESSED_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor(case, dtype):
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, index_dtype)
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, index_dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _NO_SIZE_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_no_size(case, dtype):
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(layout, shape, nnz, dtype, index_dtype)
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        compressed,
        plain,
        values,
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, index_dtype)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _EMPTY_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_empty(case, dtype):
    layout, shape = case
    compressed, plain, values = _make_input(layout, shape, 0, dtype)
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, torch.int64)
