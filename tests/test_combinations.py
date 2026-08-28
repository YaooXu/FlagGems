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

# aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
# returns a 2-D tensor whose rows are all length-r combinations of the elements
# of a 1-D input (one combination per row). Without replacement there are
# C(n, r) rows and with replacement C(n + r - 1, r) rows; r=1 returns column
# vectors of shape (n, 1) and r > n (no replacement) yields an empty (0, r)
# result. The op is a pure gather of input elements (no arithmetic), so it works
# for every storage dtype and the output dtype always matches the input dtype.
_COMBINATIONS_SHAPES = (
    [(4,), (8,)] if utils.QUICK_MODE else [(1,), (2,), (4,), (8,), (16,), (64,)]
)

_COMBINATIONS_R = [1, 2, 3]

_COMBINATIONS_WITH_REPLACEMENT = [False, True]

_FLOAT_DTYPES = set(utils.ALL_FLOAT_DTYPES)
_COMBINATIONS_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, dtype):
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.BOOL_TYPES:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)


def _resolve_gems_op():
    # Resolution order: (1) the process-local override installed by KernelGen
    # via flag_gems.testing.override_gems_op, (2) the direct
    # flag_gems.combinations callable once it is registered, (3) None -> the
    # test falls back to the PyTorch reference so it stays runnable before a
    # FlagGems implementation is registered.
    try:
        return flag_gems.testing.resolve_gems_op(
            "combinations", getattr(flag_gems, "combinations", None)
        )
    except LookupError:
        return None


def _combinations_op():
    gems_op = _resolve_gems_op()
    if gems_op is None:
        return torch.ops.aten.combinations
    return gems_op


def _assert_match(res_out, ref_out, dtype):
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    if dtype in _FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", _COMBINATIONS_SHAPES)
@pytest.mark.parametrize("r", _COMBINATIONS_R)
@pytest.mark.parametrize("with_replacement", _COMBINATIONS_WITH_REPLACEMENT)
@pytest.mark.parametrize("dtype", _COMBINATIONS_DTYPES)
def test_combinations(shape, r, with_replacement, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [1, 2, 3])
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_combinations_empty_input(r, with_replacement, dtype):
    # An empty 1-D input has no elements to combine: aten returns an empty
    # (0, r) tensor of the input dtype for every r / with_replacement setting.
    inp = _make_input((0,), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.combinations(ref_inp, r, with_replacement)
    res_out = _combinations_op()(inp, r, with_replacement)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", _COMBINATIONS_DTYPES)
def test_combinations_non_contiguous(dtype):
    # combinations gathers input elements by index, so the candidate must read
    # through the input's actual strides. Slice on both the test device and the
    # reference device so the two inputs share the same memory layout.
    base = _make_input((32,), dtype)
    ref_base = utils.to_reference(base)
    inp = base[::2]
    ref_inp = ref_base[::2]

    ref_out = torch.ops.aten.combinations(ref_inp, 2, False)
    res_out = _combinations_op()(inp, 2, False)

    _assert_match(res_out, ref_out, dtype)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_combinations_raises_on_non_1d(dtype):
    # aten::combinations only accepts 1-D inputs; the candidate must reject
    # higher-rank tensors with the same RuntimeError.
    inp = _make_input((4, 4), dtype)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, 2, False)
    gems_op = _combinations_op()
    with pytest.raises(RuntimeError):
        gems_op(inp, 2, False)


@pytest.mark.combinations
def test_combinations_raises_on_negative_r():
    # r must be non-negative; aten raises RuntimeError and the candidate must
    # behave the same way.
    inp = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.combinations(ref_inp, -1, False)
    gems_op = _combinations_op()
    with pytest.raises(RuntimeError):
        gems_op(inp, -1, False)
