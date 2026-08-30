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

# ``_values_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._values_copy`` and ``-m
# _values_copy`` both work.
setattr(
    pytest.mark,
    "_values_copy",
    MarkDecorator(Mark("_values_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_values_copy_out",
    MarkDecorator(Mark("_values_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_values_copy(Tensor self) -> Tensor materializes the (nnz,) + dense_shape
# values tensor of a sparse COO tensor as a fresh, contiguous, independent copy
# (the view_copy counterpart of aten::_values, whose native body is
# `_values(self).clone(contiguous)`). Every workload feeds a sparse COO tensor
# and checks copy semantics: the result must equal the stored values in storage
# order, must NOT alias the input's internal values storage, and must not mutate
# the input. Each (shape, sparse_dim, nnz) triple is a distinct layout: 1-D
# all-sparse, 2-D/3-D all-sparse, and mixed sparse+dense ranks up to 6-D, with
# nnz spanning the minimal (1), partial, and full sparse-volume cases so the
# (nnz,) + dense_shape shape of the result is exercised.
#
# No broadcast/backward dimensions apply: the operator is unary and returns a
# plain copy of the input's values storage (there is nothing to broadcast
# against, and `_values` on a sparse tensor does not participate in autograd in
# the installed build, so no backward test is possible).
_VALUES_COPY_CASES = [
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

# The result keeps the storage dtype of the values tensor, so the candidate must
# accept every dtype the sparse COO runtime supports: every float, int, and bool
# family.
_VALUES_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES

# Value-range coverage uses float + int storage dtypes (bool ignores the range
# in tu.make_input and adds nothing beyond the copy-semantics tests above).
_VALUE_RANGE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

# nan / +-inf stored values for the copy-semantics test.
_NAN_INF_DTYPES = utils.ALL_FLOAT_DTYPES


def _coo_value_range_cases():
    """Representative all-sparse + hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2, 8)]
    if tu.LEVEL in ("all", "extended"):
        return [((3, 4), 2, 7), ((3, 4, 2), 2, 12), ((12, 9, 3, 6), 4, 48)]
    return [((3, 4), 2, 7), ((3, 4, 2), 2, 12)]


def _make_input(shape, sparse_dim, nnz, dtype, value_range=("-1", "1"), seed=0):
    # Deterministic CPU-side index generation, then the sparse tensor is created
    # on the test device. Indices are drawn with replacement: duplicate indices
    # are allowed and merely leave the tensor uncoalesced (covered explicitly
    # below). The stored values come from the value-range framework
    # (tu.make_input) so every per-dtype range -- including the extreme
    # [min, 0] / [0, max] magnitudes -- can be exercised through this one
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


def _reference_values_copy(inp):
    # Prefer the literal ATen op as the reference. The installed PyTorch build
    # registers _values_copy as CompositeExplicitAutogradNonFunctional, whose
    # dispatch-key set excludes the Sparse functionality key, so calling
    # torch.ops.aten._values_copy directly on a sparse tensor raises
    # NotImplementedError. In that case fall back to the operator's exact
    # native body -- _values(self).clone(contiguous) -- composed from ATen
    # ops, which IS reachable on sparse tensors.
    #
    # The KernelGen ref-vs-ref verification overrides the candidate
    # (resolve_gems_op) with this same function so both sides run the same
    # native body.
    try:
        return torch.ops.aten._values_copy(inp)
    except NotImplementedError:
        return torch.ops.aten._values(inp).clone(memory_format=torch.contiguous_format)


def _reference_values_copy_out(inp, out):
    # Same strategy as _reference_values_copy for the .out overload: compute
    # the materialized copy and write it into out (the .out contract returns
    # out itself).
    try:
        return torch.ops.aten._values_copy.out(inp, out=out)
    except NotImplementedError:
        computed = torch.ops.aten._values(inp).clone(
            memory_format=torch.contiguous_format
        )
        torch.ops.aten.copy_(out, computed)
        return out


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._values_copy is registered; resolution
    # order is: (1) override, (2) the direct flag_gems._values_copy callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_values_copy", getattr(flag_gems, "_values_copy", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_values_copy.out", getattr(flag_gems, "_values_copy_out", None)
    )


def _assert_copy_semantics(res, ref, inp, ref_inp):
    # _values_copy returns a fresh contiguous (nnz,) + dense_shape tensor of the
    # input's storage dtype holding the input's stored values, in storage order
    # (nan/inf included, hence equal_nan=True). The result must not alias the
    # input's internal values storage and the input must not be mutated.
    assert res.dtype == inp.dtype
    assert ref.dtype == ref_inp.dtype
    assert res.shape == (inp._nnz(),) + inp.shape[inp.sparse_dim() :]
    assert ref.shape == (ref_inp._nnz(),) + ref_inp.shape[ref_inp.sparse_dim() :]
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref, equal_nan=True)
    # Copy semantics: fresh storage, never a view of the input's values.
    # Zero-size results (nnz == 0) may share the null pointer, so only assert
    # on non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != inp._values().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set). equal_nan=True keeps the
    # non-mutation check valid for inputs whose stored values contain nan /
    # +-inf (test__values_copy_nan_inf): a mutated tensor still differs from
    # the snapshot on the finite entries.
    utils.gems_assert_equal(inp, ref_inp, equal_nan=True)


@pytest.mark._values_copy
@pytest.mark.parametrize("case", _VALUES_COPY_CASES)
@pytest.mark.parametrize("dtype", _VALUES_COPY_DTYPES)
def test__values_copy(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._values_copy_out
@pytest.mark.parametrize("case", _VALUES_COPY_CASES)
@pytest.mark.parametrize("dtype", _VALUES_COPY_DTYPES)
def test__values_copy_out(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype)
    ref_inp = utils.to_reference(inp.clone())
    values_shape = (inp._nnz(),) + inp.shape[inp.sparse_dim() :]
    out = torch.empty(values_shape, dtype=dtype, device=inp.device)
    ref_out = torch.empty(values_shape, dtype=dtype, device=ref_inp.device)

    ref_ret = _reference_values_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark._values_copy
@pytest.mark.parametrize("dtype", _VALUES_COPY_DTYPES)
def test__values_copy_empty(dtype):
    # nnz == 0 with a non-empty dense block: _values_copy must still return a
    # (0,) + dense_shape contiguous dense tensor of the storage dtype (not a
    # sparse tensor).
    shape, sparse_dim = (3, 4, 5), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 5, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._values_copy_out
@pytest.mark.parametrize("dtype", _VALUES_COPY_DTYPES)
def test__values_copy_out_empty(dtype):
    shape, sparse_dim = (3, 4, 5), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 5, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty((0, 5), dtype=dtype, device=inp.device)
    ref_out = torch.empty((0, 5), dtype=dtype, device=ref_inp.device)

    ref_ret = _reference_values_copy_out(ref_inp, ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    assert res_ret is out
    assert ref_ret is ref_out
    _assert_copy_semantics(out, ref_out, inp, ref_inp)


@pytest.mark._values_copy
@pytest.mark.parametrize("dtype", _VALUES_COPY_DTYPES)
def test__values_copy_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _values_copy must still
    # return exactly the stored values tensor, in storage order, as an
    # independent copy (never a coalesced/sorted tensor and never an alias).
    # The (0, 1) coordinate is repeated three times and the entries are NOT
    # sorted, so a coalescing implementation would visibly change the result.
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = tu.make_input(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._values_copy
@pytest.mark.parametrize("case", _coo_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_RANGE_DTYPES)
def test__values_copy_value_ranges(case, value_range, dtype):
    # Value-range coverage from the regular-operator spec: the stored values
    # sweep the full spec range set (positive, negative, extreme and
    # degenerate); _values_copy must copy them verbatim, bit-for-bit, for every
    # layout and dtype.
    shape, sparse_dim, nnz = case
    inp = _make_input(shape, sparse_dim, nnz, dtype, value_range=value_range)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._values_copy
@pytest.mark.parametrize("dtype", _NAN_INF_DTYPES)
def test__values_copy_nan_inf(dtype):
    # nan / +-inf / +-0.0 are ordinary stored values: _values_copy must return
    # them verbatim (equal_nan=True), never sanitized.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark._values_copy
def test__values_copy_negative_dense():
    # _values_copy is a sparse-COO-only metadata accessor: a dense tensor has
    # no values storage, so both the reference and the candidate must reject
    # it. The ATen reference raises NotImplementedError (a RuntimeError
    # subclass) because the operator is only dispatched to the Sparse backends.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_values_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._values_copy
def test__values_copy_negative_csr():
    # Only sparse COO tensors expose a (nnz,) + dense_shape values tensor; a
    # sparse CSR tensor stores values in a different layout and must be
    # rejected as well.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_values_copy(utils.to_reference(inp.clone()))
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._values_copy
def test__values_copy_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises.
    with pytest.raises(RuntimeError):
        _reference_values_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve_gems_op()(3.14)


@pytest.mark._values_copy_out
def test__values_copy_out_negative_dense():
    # The .out variant is equally sparse-COO-only: a dense tensor has no values
    # storage and must be rejected.
    inp = torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(0, 0, dtype=torch.float32, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_values_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark._values_copy_out
def test__values_copy_out_negative_csr():
    # The .out variant is equally sparse-COO-only: a CSR tensor is rejected.
    crow = torch.tensor([0, 2, 3], dtype=torch.long)
    cols = torch.tensor([0, 1, 2], dtype=torch.long)
    values = torch.randn(3, dtype=torch.float32)
    inp = torch.sparse_csr_tensor(crow, cols, values, (2, 3), device=flag_gems.device)
    out = torch.empty(0, 0, dtype=torch.float32, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        _reference_values_copy_out(utils.to_reference(inp.clone()), out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)


@pytest.mark._values_copy_out
def test__values_copy_out_negative_wrong_shape():
    # The .out contract materializes exactly (nnz,) + dense_shape entries of
    # the storage dtype into out; an out tensor of the wrong shape must be
    # rejected.
    inp = _make_input((3, 4), 2, 5, torch.float32)
    ref_inp = utils.to_reference(inp.clone())
    out = torch.empty((1, 1), dtype=torch.float32, device=inp.device)
    ref_out = torch.empty((1, 1), dtype=torch.float32, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        _reference_values_copy_out(ref_inp, ref_out)
    with pytest.raises((RuntimeError, TypeError)):
        _resolve_gems_op_out()(inp, out=out)
