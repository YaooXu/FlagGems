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

# ``_nested_tensor_size`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._nested_tensor_size`` and
# ``-m _nested_tensor_size`` both work.
setattr(
    pytest.mark,
    "_nested_tensor_size",
    MarkDecorator(Mark("_nested_tensor_size", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_nested_tensor_size(Tensor self) -> Tensor returns the
# (num_tensors, num_dims) int64 sizes metadata tensor of a nested tensor.
# torch always creates that metadata on the CPU (even when the nested tensor
# lives on CUDA, see torch.nested.nested_tensor) and the native reference reads
# it with a host pointer, so the output values are exact int64 and every
# workload below compares with gems_assert_equal. Only dim 0 may be ragged;
# all other dims must match across components (ragged dims beyond 0 are a
# torch-side restriction, not a candidate property).
#
# num_tensors is kept >= 1: the reference in current torch builds segfaults on
# an empty batch (nested_tensor([])), which is a torch-side limit, not a
# candidate property.
_NUM_TENSORS = [1, 8, 256]
_NUM_DIMS = [1, 2, 3, 5]
_COMPONENT_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(num_tensors, num_dims, dtype, seed=0):
    """Deterministic strided-layout nested tensor of ``num_tensors`` rank
    ``num_dims`` components on the test device.

    Dim 0 is ragged (each component has a length in [1, 8]); every other dim is
    fixed at 4. The values are never read by the operator, but the component
    dtype is varied so the candidate must accept every storage dtype the
    nested-tensor runtime supports.
    """
    gen = torch.Generator("cpu").manual_seed(seed)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen).tolist()
    components = []
    for length in lengths:
        size = (length,) + (4,) * (num_dims - 1)
        if dtype.is_floating_point:
            components.append(torch.randn(size, dtype=dtype, generator=gen))
        elif dtype == torch.bool:
            components.append(torch.randint(0, 2, size, dtype=dtype, generator=gen))
        else:
            components.append(torch.randint(-5, 6, size, dtype=dtype, generator=gen))
    return torch.nested.nested_tensor(components, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._nested_tensor_size is registered;
    # resolution order is: (1) override, (2) the direct flag_gems callable,
    # (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_nested_tensor_size",
        getattr(flag_gems, "_nested_tensor_size", None),
    )


def _assert_sizes(res_out, ref_out, num_tensors, num_dims):
    # The sizes metadata is always a CPU int64 tensor of shape
    # (num_tensors, num_dims) and its values are exact.
    assert isinstance(res_out, torch.Tensor)
    assert isinstance(ref_out, torch.Tensor)
    assert res_out.dtype == torch.int64
    assert ref_out.dtype == torch.int64
    assert res_out.shape == (num_tensors, num_dims)
    assert ref_out.shape == (num_tensors, num_dims)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._nested_tensor_size
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
@pytest.mark.parametrize("num_dims", _NUM_DIMS)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size(num_tensors, num_dims, dtype):
    inp = _make_input(num_tensors, num_dims, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_size(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_sizes(res_out, ref_out, num_tensors, num_dims)


@pytest.mark._nested_tensor_size
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_uniform(dtype):
    # Every component has the same shape: the nested tensor is not ragged, but
    # _nested_tensor_size must still return the (num_tensors, num_dims) sizes
    # tensor rather than a strided-tensor shape.
    num_tensors, num_dims = 6, 3
    gen = torch.Generator("cpu").manual_seed(0)
    components = []
    for _ in range(num_tensors):
        if dtype.is_floating_point:
            components.append(torch.randn(4, 4, 4, dtype=dtype, generator=gen))
        elif dtype == torch.bool:
            components.append(
                torch.randint(0, 2, (4, 4, 4), dtype=dtype, generator=gen)
            )
        else:
            components.append(
                torch.randint(-5, 6, (4, 4, 4), dtype=dtype, generator=gen)
            )
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    assert inp.is_nested
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_size(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_sizes(res_out, ref_out, num_tensors, num_dims)


@pytest.mark._nested_tensor_size
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_with_empty_components(dtype):
    # Some sub-tensors are empty (dim 0 length 0); their sizes row is all
    # zeros and the running batch is still fully described.
    num_tensors, num_dims = 4, 2
    gen = torch.Generator("cpu").manual_seed(1)
    components = []
    for _ in range(num_tensors):
        length = int(torch.randint(0, 4, (1,), generator=gen).item())
        size = (length, 4)
        if dtype.is_floating_point:
            components.append(torch.randn(size, dtype=dtype, generator=gen))
        elif dtype == torch.bool:
            components.append(torch.randint(0, 2, size, dtype=dtype, generator=gen))
        else:
            components.append(torch.randint(-5, 6, size, dtype=dtype, generator=gen))
    inp = torch.nested.nested_tensor(components, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._nested_tensor_size(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_sizes(res_out, ref_out, num_tensors, num_dims)
