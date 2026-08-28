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

# aten::cartesian_prod(Tensor[] tensors) -> Tensor takes 1-D tensors and returns
# one row per combination of one element from each input. With a single input
# this PyTorch version returns the 1-D tensor itself (shape (N,)); with k
# inputs the output is (prod(sizes), k). The op performs no arithmetic: values,
# dtype and layout are copied verbatim, so every supported dtype must
# round-trip bit-exactly and the inputs must not be mutated. Each case below is
# the list of 1-D input sizes.
CARTESIAN_PROD_SIZES = (
    [[8], [3, 5], [2, 4, 3]]
    if utils.QUICK_MODE
    else [
        [8],  # single input -> (8,)
        [1],  # single singleton input -> (1,)
        [0],  # single empty input -> (0,)
        [3, 5],  # two inputs -> (15, 2)
        [16, 16],  # two equal-length inputs -> (256, 2)
        [1, 7],  # singleton + non-singleton -> (7, 2)
        [64, 128],  # larger two-input case -> (8192, 2)
        [2, 4, 3],  # three inputs -> (24, 3)
        [3, 1, 3],  # mixed singleton dims -> (9, 3)
        [8, 16, 32],  # larger three-input case -> (4096, 3)
        [2, 5, 8, 3],  # four inputs -> (240, 4)
        [0, 3],  # empty first input -> (0, 2)
        [5, 0],  # empty second input -> (0, 2)
    ]
)

_FLOAT_DTYPES = set(utils.ALL_FLOAT_DTYPES)
CARTESIAN_PROD_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(size, dtype):
    if dtype in _FLOAT_DTYPES:
        return torch.randn(size, dtype=dtype, device=flag_gems.device)
    if dtype in utils.BOOL_TYPES:
        return torch.randint(0, 2, (size,), dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, (size,), dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolution order: (1) the process-local override installed by KernelGen
    # via flag_gems.testing.override_gems_op, (2) the direct
    # flag_gems.cartesian_prod callable once it is registered, (3) None -> the
    # test falls back to the PyTorch reference so it stays runnable before a
    # FlagGems implementation is registered.
    try:
        return flag_gems.testing.resolve_gems_op(
            "cartesian_prod", getattr(flag_gems, "cartesian_prod", None)
        )
    except LookupError:
        return None


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", CARTESIAN_PROD_SIZES)
@pytest.mark.parametrize("dtype", CARTESIAN_PROD_DTYPES)
def test_cartesian_prod(sizes, dtype):
    inp = [_make_input(size, dtype) for size in sizes]
    inp_before = [t.clone() for t in inp]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.cartesian_prod(ref_inp)
    gems_op = _resolve_gems_op()
    if gems_op is None:
        res_out = torch.ops.aten.cartesian_prod(inp)
    else:
        res_out = gems_op(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    # cartesian_prod is a pure gather: the inputs must not be mutated.
    for t, before in zip(inp, inp_before):
        utils.gems_assert_equal(t, utils.to_reference(before))
