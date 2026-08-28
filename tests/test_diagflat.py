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

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# aten::diagflat(Tensor self, int offset=0) -> Tensor flattens ``self`` into a
# 1-D vector (in logical row-major view order) and returns a 2-D matrix whose
# ``offset`` diagonal holds that vector while every other entry is zero. The
# output is square with side length ``numel(self) + abs(offset)``, so the
# shapes below keep the input element count small enough that the quadratic
# output stays reasonable for correctness runs.
DIAGFLAT_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [
        (),
        (1,),
        (16,),
        (257,),
        (0,),
        (2, 3),
        (32, 32),
        (4, 5, 6),
        (2, 3, 4, 5),
        (2, 2, 2, 2, 3),
    ]
)

DIAGFLAT_OFFSETS = [-2, -1, 0, 1, 2]

# diagflat is a pure data-movement op, so every storage dtype aten supports is
# exercised: float (incl. float64 when the device supports it), int, and bool.
DIAGFLAT_DTYPES = utils.ALL_FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.BOOL_TYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    return torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. flag_gems.diagflat is
    # not part of the public namespace yet, so the default stays None until it
    # exists; without an override the resolution raises LookupError.
    return flag_gems.testing.resolve_gems_op(
        "diagflat", getattr(flag_gems, "diagflat", None)
    )


def _assert_output(res_out, ref_out, dtype):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", DIAGFLAT_SHAPES)
@pytest.mark.parametrize("offset", DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", DIAGFLAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(2,), (16,)])
@pytest.mark.parametrize("offset", [-7, -3, 3, 7])
@pytest.mark.parametrize("dtype", DIAGFLAT_DTYPES)
def test_diagflat_large_offset(shape, offset, dtype):
    # Offsets whose magnitude may exceed the number of elements: the flattened
    # vector must be placed on a diagonal that starts past the main diagonal,
    # leaving extra zero rows/columns around it.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", [(4, 8), (6, 3), (2, 3, 4)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", DIAGFLAT_DTYPES)
def test_diagflat_non_contiguous(shape, offset, dtype):
    # diagflat flattens the logical view, so a transposed (non-contiguous)
    # input must produce a different diagonal order than a contiguous one.
    # Slice on both the test device and the reference device so the two inputs
    # share the same memory layout.
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    inp = inp.transpose(-1, -2)
    ref_inp = ref_inp.transpose(-1, -2)

    ref_out = torch.ops.aten.diagflat(ref_inp, offset)
    res_out = _resolve_gems_op()(inp, offset)

    _assert_output(res_out, ref_out, dtype)
