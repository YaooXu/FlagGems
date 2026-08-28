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

# The KernelGen verification harness stages this file in a temporary tree and
# runs pytest in-process with --import-mode=importlib, where the checkout root
# is not on sys.path (a `python -m pytest` invocation would normally place it
# there). Insert the checkout root so the sibling accuracy_utils package below
# resolves identically in the in-tree and staged verification layouts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.mark.structures import Mark, MarkDecorator  # noqa: E402

import flag_gems  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

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
# all-sparse, 2-D/3-D all-sparse, and mixed sparse+dense ranks up to 5-D, with
# varying nnz so the (nnz,) + dense_shape shape of the result is exercised.
_VALUES_COPY_CASES = [
    ((5,), 1, 4),
    ((3, 4), 2, 7),
    ((3, 4), 1, 16),
    ((8, 8, 8), 3, 32),
    ((3, 4, 2), 2, 12),
    ((4, 3, 4, 5), 1, 24),
    ((3, 4, 5, 4, 5), 3, 40),
]

# The result keeps the storage dtype of the values tensor, so the candidate must
# accept every dtype the sparse COO runtime supports: every float, int, and bool
# family.
_VALUES_COPY_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


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
    return torch.ops.aten._values(inp).clone(memory_format=torch.contiguous_format)


def _reference_values_copy_out(inp, out):
    # Same strategy as _reference_values_copy for the .out overload: compute
    # the materialized copy and write it into out (the .out contract returns
    # out itself).
    computed = torch.ops.aten._values(inp).clone(memory_format=torch.contiguous_format)
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
    # input's storage dtype holding the input's stored values, in storage order.
    # The result must not alias the input's internal values storage and the
    # input must not be mutated.
    assert res.dtype == inp.dtype
    assert ref.dtype == ref_inp.dtype
    assert res.shape == (inp._nnz(),) + inp.shape[inp.sparse_dim() :]
    assert ref.shape == (ref_inp._nnz(),) + ref_inp.shape[ref_inp.sparse_dim() :]
    assert res.is_contiguous()
    utils.gems_assert_equal(res, ref)
    # Copy semantics: fresh storage, never a view of the input's values.
    # Zero-size results (nnz == 0) may share the null pointer, so only assert
    # on non-empty results.
    if res.numel() > 0:
        assert res.data_ptr() != inp._values().data_ptr()
    # The accessor must not mutate the input: ref_inp is a pre-call snapshot
    # (a clone, moved to CPU when TO_CPU is set).
    utils.gems_assert_equal(inp, ref_inp)


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

    ref_out = _reference_values_copy(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)
