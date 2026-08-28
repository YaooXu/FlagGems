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

# ``_nested_compute_contiguous_strides_offsets`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so
# ``@pytest.mark._nested_compute_contiguous_strides_offsets`` and
# ``-m _nested_compute_contiguous_strides_offsets`` both work.
setattr(
    pytest.mark,
    "_nested_compute_contiguous_strides_offsets",
    MarkDecorator(
        Mark("_nested_compute_contiguous_strides_offsets", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_nested_compute_contiguous_strides_offsets(Tensor nested_size)
# -> (Tensor, Tensor) derives, from the sizes of every sub-tensor of a nested
# tensor, the contiguous strides and the storage offsets of each sub-tensor.
# ``nested_size`` is the (num_tensors, num_dims) int64 sizes tensor that nested
# tensors carry: torch itself always creates it on the CPU (even when the
# nested tensor lives on CUDA, see torch.nested.as_nested_tensor), and the
# native reference reads the int64 data with a host pointer, so every workload
# below feeds a CPU int64 tensor and compares the two int64 outputs exactly.
#
# Layout semantics:
#   strides[i][j] = prod(sizes[i][j+1:])      (contiguous row-major strides)
#   offsets[0]    = 0
#   offsets[i]    = offsets[i-1] + prod(sizes[i-1])
#
# num_tensors is kept >= 1: the reference in current torch builds segfaults on
# an empty batch (nested_size with numel() == 0), which is a torch-side limit,
# not a candidate property.
_NUM_TENSORS = [1, 3, 8, 64, 512]
_NUM_DIMS = [1, 2, 3, 5]
_SIZES_PATTERNS = ["uniform", "with_zero", "all_ones", "wide"]


def _make_nested_size(num_tensors, num_dims, pattern, seed=0):
    """Deterministic (num_tensors, num_dims) int64 sizes tensor per pattern."""
    gen = torch.Generator("cpu").manual_seed(seed)
    if pattern == "uniform":
        # Typical ragged batch: every sub-tensor is non-empty with varied dims.
        return torch.randint(
            1, 9, (num_tensors, num_dims), dtype=torch.int64, generator=gen
        )
    if pattern == "with_zero":
        # Some sub-tensors are empty (zero-size dims); offsets may repeat.
        return torch.randint(
            0, 9, (num_tensors, num_dims), dtype=torch.int64, generator=gen
        )
    if pattern == "all_ones":
        # Minimal positive sizes: every product is 1.
        return torch.ones((num_tensors, num_dims), dtype=torch.int64)
    if pattern == "wide":
        # Wider value range than "uniform". The bound keeps every per-row
        # product below 2**20 (in the worst case 2**(20//num_dims) ** num_dims):
        # the reference in current torch builds truncates each per-row product
        # to int32 when it reaches 2**31 (offsets[i-1] + sizes[i-1]*strides[i-1]
        # wraps, e.g. a single size of 2**31 yields a negative offset), so the
        # sizes must stay well under that limit for the reference itself to be
        # correct. The running offset sum over the batch (<= 512 rows) then
        # stays below 512 * 2**20 = 2**29, far inside int64.
        bound = max(2, 2 ** (20 // num_dims))
        return torch.randint(
            1, bound, (num_tensors, num_dims), dtype=torch.int64, generator=gen
        )
    raise ValueError(f"Unknown size pattern: {pattern!r}")


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._nested_compute_contiguous_strides_offsets
    # is registered; resolution order is: (1) override, (2) the direct
    # flag_gems callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_nested_compute_contiguous_strides_offsets",
        getattr(flag_gems, "_nested_compute_contiguous_strides_offsets", None),
    )


def _assert_layout(num_tensors, num_dims, res_strides, res_offsets):
    # Structural checks plus layout invariants that hold for any correct
    # implementation, independent of the reference: innermost stride is always
    # 1, the first sub-tensor starts at offset 0 and (sizes are non-negative)
    # the offsets never go backwards.
    assert isinstance(res_strides, torch.Tensor)
    assert isinstance(res_offsets, torch.Tensor)
    assert res_strides.dtype == torch.int64
    assert res_offsets.dtype == torch.int64
    assert res_strides.shape == (num_tensors, num_dims)
    assert res_offsets.shape == (num_tensors,)
    assert bool(torch.all(res_strides[:, -1] == 1))
    assert int(res_offsets[0]) == 0
    assert bool(torch.all(res_offsets[1:] >= res_offsets[:-1]))


@pytest.mark._nested_compute_contiguous_strides_offsets
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
@pytest.mark.parametrize("num_dims", _NUM_DIMS)
@pytest.mark.parametrize("pattern", _SIZES_PATTERNS)
def test__nested_compute_contiguous_strides_offsets(num_tensors, num_dims, pattern):
    sizes = _make_nested_size(num_tensors, num_dims, pattern)
    ref_sizes = utils.to_reference(sizes)

    ref_strides, ref_offsets = (
        torch.ops.aten._nested_compute_contiguous_strides_offsets(ref_sizes)
    )
    res_strides, res_offsets = _resolve_gems_op()(sizes)

    _assert_layout(num_tensors, num_dims, res_strides, res_offsets)
    utils.gems_assert_equal(res_strides, ref_strides)
    utils.gems_assert_equal(res_offsets, ref_offsets)
