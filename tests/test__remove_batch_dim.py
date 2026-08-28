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

# The correctness suite is imported as the top-level ``tests`` package. The
# runner may already have a *different* ``tests`` package importable on
# sys.path (e.g. the test-writer harness's own ``kernelgen/tests``), and pytest
# binds the parent package before executing this module. Force the resolution
# to the ``tests`` package that ships with this file so the relative import
# below is unambiguous.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_TEST_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
_IMPORTED_TESTS = sys.modules.get("tests")
if _IMPORTED_TESTS is not None and os.path.abspath(
    getattr(_IMPORTED_TESTS, "__file__", "")
) != os.path.join(_TEST_DIR, "__init__.py"):
    del sys.modules["tests"]

from . import accuracy_utils as utils  # noqa: E402

# ``_remove_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._remove_batch_dim`` and ``-m
# _remove_batch_dim`` both work.
setattr(
    pytest.mark,
    "_remove_batch_dim",
    MarkDecorator(Mark("_remove_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_remove_batch_dim(Tensor self, int level, SymInt batch_size, int out_dim)
# is the functorch/vmap unwrap primitive. On a plain tensor it is exactly
# ``self.expand(sizes)`` where ``sizes`` is ``self.shape`` with ``batch_size``
# inserted at position ``out_dim``: a batch dimension of size ``batch_size`` is
# created at ``out_dim`` and broadcast along the whole tensor (the batch dim gets
# stride 0). ``level`` is only vmap bookkeeping and does not affect the result.
# Broadcast is valid when every dim of ``self`` is either equal to the
# corresponding target dim or is 1; the (shape, out_dim, batch_size) cases below
# are chosen so the expand always succeeds and cover ranks 1-5, every valid
# out_dim position, batch_size matching the adjacent dim, and size-1 broadcast.
REMOVE_BATCH_DIM_CASES = [
    ((16,), 0, 7),
    ((16,), 1, 16),
    ((64, 32), 0, 13),
    ((64, 32), 1, 64),
    ((2, 19, 7), 0, 5),
    ((2, 19, 7), 1, 2),
    ((1, 19, 7), 1, 5),
    ((4, 4, 16), 2, 4),
    ((1, 19, 7), 2, 19),
    ((4, 8, 16, 32), 0, 9),
    ((4, 8, 16, 32), 1, 4),
    ((8, 8, 8, 32), 2, 8),
    ((16, 7, 57, 32, 29), 0, 11),
    ((16, 7, 57, 32, 29), 1, 16),
]

# The op is a pure broadcast view: no arithmetic is performed, so every storage
# dtype is supported.
REMOVE_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order is:
    # (1) override, (2) the direct flag_gems._remove_batch_dim callable, (3)
    # LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_remove_batch_dim", getattr(flag_gems, "_remove_batch_dim", None)
    )


def _expected_shape(shape, out_dim, batch_size):
    sizes = list(shape)
    sizes.insert(out_dim, batch_size)
    return tuple(sizes)


def _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype):
    assert res_out.shape == ref_out.shape == _expected_shape(shape, out_dim, batch_size)
    assert res_out.dtype == ref_out.dtype
    # Broadcasting repeats the physical values exactly, so the candidate and the
    # reference must agree element-wise.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", REMOVE_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim(shape, out_dim, batch_size, level, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize(
    "shape, out_dim, batch_size",
    [((8, 16, 32), 1, 8), ((8, 8, 16, 32), 2, 8)],
)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_non_contiguous(shape, out_dim, batch_size, level, dtype):
    # The broadcast view must preserve the strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout.
    base = _make_input(shape, dtype)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, inp.shape, out_dim, batch_size, dtype)
