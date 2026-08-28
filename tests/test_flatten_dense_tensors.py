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

import sys as _sys
from pathlib import Path as _Path

import pytest
import torch

import flag_gems

# The KernelGen integration harness verifies this file inside a temporary copy
# of the FlagGems tree. That process is launched with sys.path[0] pointing at
# the harness script, not the tree root, so the parent `tests` package would
# not resolve. Insert the tree root so the relative import below always works.
_TREE_ROOT = str(_Path(__file__).resolve().parents[1])
if _TREE_ROOT not in _sys.path:
    _sys.path.insert(0, _TREE_ROOT)

from . import accuracy_utils as utils  # noqa: E402

# aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor is the DDP
# gradient-flattening utility: it flattens every input to a contiguous 1-D
# tensor (t.contiguous().view(-1)) and concatenates them into one 1-D result.
# It is a pure data-movement op (copy + cat): inputs are never mutated and all
# tensors in the list must share the same dtype and device. Each parametrized
# combination below is a distinct workload; the list-of-shapes selects the
# number, ranks, sizes and (via the dedicated non-contiguous test) the memory
# layout of the tensors fed to the op.

# Workloads: a single tensor, several same-shape tensors, mixed ranks/sizes,
# large tensors, empty tensors and 0-dim tensors. Element counts stay small
# (<= 1M) so correctness runs stay fast.
FLATTEN_DENSE_TENSORS_SHAPE_CASES = [
    [(2, 3)],
    [(4, 5), (4, 5), (4, 5)],
    [(2, 3), (4,), (5, 6, 7)],
    [(1024,), (64, 64), (16, 16, 16)],
    [(0, 3), (2,), (1, 1, 1)],
    [(), (3,), (1, 4)],
]

# The op only moves data, so every storage dtype is supported: float families
# plus integer and bool dtypes.
FLATTEN_DENSE_TENSORS_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)


def _make_input_tensors(tensor_shapes, dtype):
    """Build a same-dtype tensor list with per-dtype value ranges."""
    tensors = []
    for shape in tensor_shapes:
        if dtype.is_floating_point:
            tensors.append(torch.randn(shape, dtype=dtype, device=flag_gems.device))
        elif dtype == torch.bool:
            tensors.append(torch.rand(shape, device=flag_gems.device) < 0.5)
        else:
            tensors.append(
                torch.empty(shape, dtype=dtype, device=flag_gems.device).random_()
            )
    return tensors


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that a process-local
    # override installed by KernelGen wins. Resolution order: (1) override,
    # (2) the direct flag_gems.flatten_dense_tensors callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "flatten_dense_tensors", getattr(flag_gems, "flatten_dense_tensors", None)
    )


def _assert_flattened(res_out, ref_out, dtype):
    # The result is a 1-D tensor of the input dtype holding every input element
    # in order (contiguous copy then concatenate).
    assert res_out.dim() == 1
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == dtype
    assert res_out.numel() == ref_out.numel()
    assert res_out.device.type == torch.device(flag_gems.device).type
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", FLATTEN_DENSE_TENSORS_SHAPE_CASES)
@pytest.mark.parametrize("dtype", FLATTEN_DENSE_TENSORS_DTYPES)
def test_flatten_dense_tensors(tensor_shapes, dtype):
    tensors = _make_input_tensors(tensor_shapes, dtype)
    tensors_before = [t.clone() for t in tensors]
    ref_tensors = [utils.to_reference(t) for t in tensors]
    expected_numel = sum(t.numel() for t in tensors)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_tensors)
    res_out = _resolve_gems_op()(tensors)

    _assert_flattened(res_out, ref_out, dtype)
    assert res_out.numel() == expected_numel
    # The op is read-only: inputs must be left untouched.
    for res_t, before in zip(tensors, tensors_before):
        utils.gems_assert_equal(res_t, before)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", FLATTEN_DENSE_TENSORS_DTYPES)
def test_flatten_dense_tensors_non_contiguous(dtype):
    # Column slices and a transpose exercise the contiguous-copy path of the op.
    base = _make_input_tensors([(8, 16)], dtype)[0]
    ref_base = utils.to_reference(base)
    views = [base[:, ::2], base.t(), base.reshape(4, 32)[:, ::3]]
    ref_views = [ref_base[:, ::2], ref_base.t(), ref_base.reshape(4, 32)[:, ::3]]
    assert all(not v.is_contiguous() for v in views)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_views)
    res_out = _resolve_gems_op()(views)

    _assert_flattened(res_out, ref_out, dtype)
