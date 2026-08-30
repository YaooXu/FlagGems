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
from . import test_utils as tu  # noqa: E402

# aten::sparse_compressed_tensor.comp_plain_value_size(Tensor compressed_indices,
#     Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# and aten::sparse_compressed_tensor.comp_plain_value(Tensor compressed_indices,
#     Tensor plain_indices, Tensor values, *, ScalarType? dtype=None,
#     Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
# construct a sparse compressed tensor (CSR/CSC/BSR/BSC) from raw index tensors
# and values. The generic factory requires the layout kwarg and, unlike the
# torch.sparse_csr_tensor wrappers, does NOT infer the dtype from values: dtype=
# must be passed whenever the values dtype is not float32. On GPU the device=
# kwarg is required too (without it the instance is created on CPU and rejected
# by the same-device invariant in SparseCsrTensorImpl::set_member_tensors).
# Every workload below therefore calls both the reference and the candidate with
# dtype= and device= passed explicitly.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shape levels: (layout, shape, nnz, index_dtype) structures from the
#     quick/core/all TEST_LEVELs: 2-D CSR/CSC, 2-D block BSR/BSC (2x2 blocks),
#     and 3-D/4-D batched CSR/CSC/BSR with int64 and int32 index tensors;
#   * value ranges: tu.selected_ranges() over representative layouts, so every
#     storage dtype is exercised with negative, positive, extreme and degenerate
#     value ranges (the factory copies the values verbatim, so the structural
#     result is identical for all of them);
#   * edge cases: empty storage (nnz == 0) for every layout, size-inferred
#     (no-size) construction, and nan/inf/-inf/±0.0 stored values;
#   * backward: the factory records an autograd formula w.r.t. the values
#     input, so gradients of a scalar function of the result's values are
#     compared against the CPU reference;
#   * negative: missing layout, non-compressed (COO) layout, dtype mismatch,
#     and non-float32 values without an explicit dtype are rejected.
#
# Broadcast does not apply: the factory combines index/value component tensors
# into a single sparse structure rather than combining two same-shaped tensors
# elementwise, so there are no broadcast dimensions to exercise.

# (layout, shape, nnz, index_dtype) structures: 2-D CSR/CSC, 2-D block BSR/BSC,
# and batched CSR (int64 and int32 index tensors).
_CORE_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_csc, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
    (torch.sparse_bsc, (4, 6), 6, torch.int64),
    (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
    (torch.sparse_csr, (3, 5, 4), 7, torch.int32),
]

# Higher-rank / multi-batch-dims structures for the "all"/"extended" TEST_LEVEL.
_ALL_CASES = _CORE_CASES + [
    (torch.sparse_bsr, (2, 4, 6, 4), 8, torch.int64),
    (torch.sparse_csc, (3, 4, 5, 6), 9, torch.int32),
]

_QUICK_CASES = [
    (torch.sparse_csr, (2, 19, 7), 12, torch.int64),
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

# Representative layouts for the value-range sweep: a 2-D CSR, a block BSR, and
# a batched CSR at core; the "all" level adds a multi-batch-dim batched BSC.
_CORE_VALUE_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
    (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
]

_ALL_VALUE_CASES = _CORE_VALUE_CASES + [
    (torch.sparse_bsc, (2, 4, 4, 6), 8, torch.int64),
]

_QUICK_VALUE_CASES = [
    (torch.sparse_csr, (2, 19, 7), 12, torch.int64),
]

# The factory accepts every storage dtype the sparse compressed runtime
# supports: all float, all int, and bool.
_SPARSE_COMPRESSED_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2


def _layout_cases():
    """(layout, shape, nnz, index_dtype) structures by TEST_LEVEL."""
    if tu.LEVEL == "quick":
        return _QUICK_CASES
    if tu.LEVEL in ("all", "extended"):
        return _ALL_CASES
    return _CORE_CASES


def _value_range_cases():
    """Representative structures for the value-range sweep by TEST_LEVEL."""
    if tu.LEVEL == "quick":
        return _QUICK_VALUE_CASES
    if tu.LEVEL in ("all", "extended"):
        return _ALL_VALUE_CASES
    return _CORE_VALUE_CASES


def _make_input(
    layout, shape, nnz, dtype, index_dtype=torch.int64, value_range=None, seed=0
):
    # Deterministic CPU-side generation of a valid compressed structure; the
    # values tensor comes from the shared value-range helper (tu.make_input) and
    # both the index tensors and the values are moved to the test device.
    # (row, col) block entries are drawn with replacement and sorted, and the
    # compressed pointer array is built with a per-batch bincount, so the
    # structure is always a valid compressed sparse tensor.
    if value_range is None:
        value_range = ["-1", "1"]
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
    values = tu.make_input(dtype, entries + block_shape, value_range)
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
@pytest.mark.parametrize("case", _layout_cases())
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values
    # for every storage dtype (bool/int snap the range to the representable
    # set). Each (layout, shape, nnz, index_dtype) structure is a distinct
    # workload; the explicit size overload (4 components + size) is exercised.
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
    # Size-inferred overload: the 3-component call (no size) derives the tensor
    # shape from the index tensors and the values block shape. The inferred
    # shape must match the reference exactly.
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
    # nnz == 0 for every layout: the compressed index tensor keeps its full
    # compressed_dim + 1 entries and the values tensor is empty, but the
    # constructed tensor must still be a valid sparse compressed tensor with
    # the requested layout and dtype.
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


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_value_ranges(case, value_range, dtype):
    # The stored values sweep the full spec range set (positive, negative,
    # extreme and degenerate); the factory copies the payload verbatim, so the
    # whole constructed tensor must round-trip every range bit-for-bit
    # (tu.assert_result_close compares int/bool exactly and floats with
    # equal_nan=True).
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, index_dtype, value_range
    )
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
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_compressed_tensor_nan_inf_values(dtype):
    # nan/inf/-inf and signed zeros are ordinary stored values: the factory
    # copies them verbatim (it performs no arithmetic on the payload), so the
    # constructed tensor must preserve them bit-for-bit (equal_nan=True).
    layout = torch.sparse_csr
    shape = (3, 4)
    crow_t = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(
        [0, 1, 0, 2, 1, 2, 0], dtype=torch.long, device=flag_gems.device
    )
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5, -2.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_crow = utils.to_reference(crow_t)
    ref_col = utils.to_reference(col_t)
    ref_values = utils.to_reference(values)

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_crow.device,
    )
    gems_op = _resolve_gems_op()
    res_out = gems_op(
        crow_t,
        col_t,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=crow_t.device,
    )

    assert res_out.layout == layout
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out) == 7
    utils.gems_assert_equal(res_out.crow_indices(), ref_out.crow_indices())
    utils.gems_assert_equal(res_out.col_indices(), ref_out.col_indices())
    utils.gems_assert_equal(res_out.values(), ref_out.values(), equal_nan=True)


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_compressed_tensor_backward(dtype):
    # The factory records an autograd formula (SparseCompressedTensorBackward0)
    # w.r.t. the values input, so gradients flow back through the constructed
    # tensor's values payload. The factory copies the values verbatim, so the
    # gradient of a scalar function of the result's values w.r.t. the values
    # input equals that scalar function's local gradient; compare the candidate
    # and the reference for the same weighted sum.
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values).detach().clone().requires_grad_(True)

    values_in = values.detach().clone().requires_grad_(True)

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
        values_in,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
    )

    _assert_result(res_out, ref_out, dtype, layout, torch.int64)

    weights = torch.linspace(-1.0, 1.0, 7, dtype=dtype, device=flag_gems.device)
    grad_ref = torch.autograd.grad((ref_out.values() * weights).sum(), ref_values)[0]
    grad_res = torch.autograd.grad((res_out.values() * weights).sum(), values_in)[0]
    utils.gems_assert_close(grad_res, grad_ref, dtype)


def _negative_csr_inputs(dtype=torch.float32):
    # A valid CSR (compressed, plain, values) triple plus its logical shape,
    # reused by the negative tests that then pass invalid kwargs.
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    return compressed, plain, values, shape


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_missing_layout():
    # The generic factory dispatches on the layout kwarg; without it there is no
    # backend to construct the instance with and aten raises RuntimeError. The
    # candidate must reject the call too rather than silently defaulting to a
    # dense or COO tensor.
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_coo_layout():
    # Sparse (COO) is a distinct backend key from the sparse compressed layouts;
    # the factory has no Sparse implementation and raises. The candidate must
    # reject it too.
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_dtype_mismatch():
    # The dtype= kwarg must match the values dtype; aten raises when it does
    # not. The candidate must reproduce the validation.
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_non_float32_values_without_dtype():
    # Unlike the torch.sparse_csr_tensor wrappers, the generic factory does NOT
    # infer the dtype from the values: a bf16 values tensor without dtype= is
    # rejected because the instance defaults to float32. The candidate must
    # reject it too rather than silently upcasting the payload.
    compressed, plain, values, shape = _negative_csr_inputs(torch.bfloat16)
    ref_compressed = utils.to_reference(compressed)
    ref_plain = utils.to_reference(plain)
    ref_values = utils.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(
            compressed,
            plain,
            values,
            list(shape),
            layout=torch.sparse_csr,
            device=compressed.device,
        )
