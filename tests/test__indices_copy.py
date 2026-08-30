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
from _pytest.mark.structures import Mark, MarkDecorator

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
# 2-D/3-D all-sparse, and mixed sparse+dense ranks up to 6-D, with nnz spanning
# the minimal (1), partial, and full sparse-volume cases so the
# (sparse_dim, nnz) shape of the result is exercised.
_INDICES_COPY_CASES = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
    ((2, 3), 2, 6),
    ((6, 6), 2, 1),
    ((2, 2, 2, 2, 2, 2), 3, 16),
]

# The result ignores the stored values, but the candidate must accept any
# storage dtype the sparse COO runtime supports: every float, int, and bool
# family.
_INDICES_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage uses float + int storage dtypes (bool ignores the range
# in tu.make_input and adds nothing beyond the copy-semantics tests above).
_VALUE_RANGE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# nan / +-inf stored values for the metadata-accessor test.
_NAN_INF_DTYPES = utils.ALL_FLOAT_DTYPES


def _make_input(shape, sparse_dim, nnz, dtype, value_range=("-1", "1"), seed=0):
    # Deterministic CPU-side index generation, then the sparse tensor is created
    # on the test device. Indices are drawn with replacement: duplicate indices
    # are allowed and merely leave the tensor uncoalesced (covered explicitly
    # below). The stored values come from the value-range framework
    # (tu.make_input) because the returned index tensor never depends on them;
    # every per-dtype range can therefore be exercised through this one
    # constructor.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = tu.make_input(dtype, (nnz,) + tuple(dense_shape), list(value_range))
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
    try:
        return torch.ops.aten._indices_copy(inp)
    except NotImplementedError:
        return torch.ops.aten._indices(inp).clone(memory_format=torch.contiguous_format)


def _reference_indices_copy_out(inp, out):
    # Same strategy as _reference_indices_copy for the .out overload: compute
    # the materialized copy and write it into out (the .out contract returns
    # out itself).
    try:
        return torch.ops.aten._indices_copy.out(inp, out=out)
    except NotImplementedError:
        computed = torch.ops.aten._indices(inp).clone(
            memory_format=torch.contiguous_format
        )
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
    # (a clone, moved to CPU when TO_CPU is set). equal_nan=True keeps the
    # non-mutation check valid for inputs whose stored values contain nan /
    # +-inf (test__indices_copy_nan_inf_values): a mutated tensor still differs
    # from the snapshot on the finite entries.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


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
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test__indices_copy_value_ranges(dtype, value_range):
    # Value-range coverage from the regular-operator spec: the metadata output
    # (the (sparse_dim, nnz) int64 index tensor) is independent of the stored
    # values, so every per-dtype range -- including the extreme [min, 0] and
    # [0, max] magnitudes -- must be accepted and must not perturb the returned
    # indices.
    shape, sparse_dim, nnz = (3, 4), 2, 7
    inp = _make_input(shape, sparse_dim, nnz, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test__indices_copy_nan_inf_values(dtype):
    # nan / +-inf stored values must not perturb the returned index tensor:
    # _indices_copy reads only the index storage, so the copy must still be
    # bit-exact even when the values contain non-finite entries.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 1, 2], [0, 1, 1, 0, 3, 3]], dtype=torch.long)
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 1.5, -2.5, 0.0],
        dtype=dtype,
    )
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_indices_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._indices_copy
def test__indices_copy_negative_dense():
    # _indices_copy is a sparse-COO-only metadata accessor: a dense tensor has
    # no index storage, so both the reference and the candidate must reject it.
    # The ATen reference raises NotImplementedError (a RuntimeError subclass)
    # because the operator is only dispatched to the Sparse backends.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._indices_copy
def test__indices_copy_negative_csr():
    # Only sparse COO tensors expose an (sparse_dim, nnz) coordinate index
    # tensor; a sparse CSR tensor stores crow/col pointers and must be rejected
    # as well.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_indices_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._indices_copy_out
def test__indices_copy_out_negative_dense():
    # The .out variant is equally sparse-COO-only: a dense tensor has no index
    # storage and must be rejected.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(0, 0, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark._indices_copy_out
def test__indices_copy_out_negative_csr():
    # The .out variant is equally sparse-COO-only: a CSR tensor is rejected.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    out = torch.empty(0, 0, dtype=torch.long, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_indices_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark._indices_copy_out
def test__indices_copy_out_negative_wrong_shape():
    # The .out contract materializes exactly (sparse_dim, nnz) int64 entries
    # into out; an out tensor of the wrong shape must be rejected.
    inp = _make_input((3, 4), 2, 5, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty((1, 1), dtype=torch.long, device=inp.device)
    ref_out = torch.empty((1, 1), dtype=torch.long, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_indices_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
