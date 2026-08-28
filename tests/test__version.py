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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

# ``_version`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._version`` and ``-m _version`` both
# work.
setattr(
    pytest.mark,
    "_version",
    MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_version(Tensor self) -> int reports the version counter of a tensor
# (the number of in-place mutations applied to it so far). It is a pure
# metadata query whose result never depends on the shape, layout, or dtype of
# the storage. The shapes below are deliberately small because the op is O(1),
# but they cover every rank from 0-D to 5-D so the candidate must accept
# arbitrary tensors.
_VERSION_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (3, 4), (8, 16, 4), (2, 3, 4, 5), (4, 7, 5, 3, 2)]
)

# The version counter lives on the TensorImpl, so the storage dtype is
# irrelevant; still, cover every dtype family the runtime supports.
_VERSION_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    # _version never touches the tensor data, so zeros are sufficient.
    return torch.zeros(shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # ``flag_gems._version`` is the package version module (package metadata),
    # not the operator callable, so the default is left as None and resolution
    # order stays: (1) the process-local override injected by KernelGen, (2) a
    # registered ``flag_gems._version`` operator callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_version", None)


def _assert_result(res_out, ref_out):
    # _version returns a plain Python int holding the version counter, so exact
    # equality is required and no tolerance is involved.
    assert type(res_out) is int
    assert type(ref_out) is int
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_fresh(shape, dtype):
    # A freshly created tensor starts at version 0.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert res_out == 0


@pytest.mark._version
@pytest.mark.parametrize("dtype", _VERSION_DTYPES)
def test__version_dtype_coverage(dtype):
    # The version counter ignores the storage dtype; every dtype family must be
    # accepted and reported as 0 for a fresh tensor.
    inp = _make_input((3, 4), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert res_out == 0


@pytest.mark._version
@pytest.mark.parametrize("shape", _VERSION_SHAPES)
@pytest.mark.parametrize("bumps", [1, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_after_inplace(shape, bumps, dtype):
    # Every in-place mutation increments the version counter by one; the op
    # must report the exact number of mutations applied to the tensor. The
    # reference input may be the same tensor (CPU reference disabled) or an
    # independent copy (TO_CPU), so bump it exactly ``bumps`` times either way.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    for _ in range(bumps):
        torch.ops.aten.add_.Tensor(inp, 1)
    if ref_inp is not inp:
        for _ in range(bumps):
            torch.ops.aten.add_.Tensor(ref_inp, 1)

    ref_out = torch.ops.aten._version(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out)
    assert res_out == bumps


@pytest.mark._version
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_readonly(dtype):
    # _version is a read-only query: it must neither bump the version counter
    # nor modify the tensor data.
    inp = _make_input((8, 16), dtype)
    ref_inp = utils.to_reference(inp)
    data_before = inp.clone()
    version_before = torch.ops.aten._version(ref_inp)

    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, version_before)
    assert torch.ops.aten._version(inp) == version_before
    assert torch.equal(inp, data_before)


@pytest.mark._version
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__version_view(dtype):
    # Views share the version counter with their base tensor, so the op must
    # report the same value for a view as for the base.
    inp = _make_input((4, 6), dtype)
    ref_inp = utils.to_reference(inp)
    view = inp.view(3, 8)
    ref_view = ref_inp.view(3, 8)

    ref_base = torch.ops.aten._version(ref_inp)
    ref_out = torch.ops.aten._version(ref_view)
    res_out = _resolve_gems_op()(view)

    _assert_result(res_out, ref_out)
    assert res_out == ref_base
