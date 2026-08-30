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
# stride 0). ``level`` is only vmap bookkeeping and never affects the result.
# Broadcast is valid when every dim of ``self`` (aligned to the trailing dims of
# the target) is either equal to the corresponding target dim or is 1, so the
# (shape, out_dim, batch_size) cases below are chosen so the expand always
# succeeds. Together they cover ranks 0-5, every valid out_dim class (front,
# middle, end), batch_size matching the adjacent dim, and size-1 broadcast. The
# op performs no arithmetic (it is a pure zero-copy view), so every storage
# dtype is supported and element counts stay small (the output is a stride-0
# view, only the assert materializes it).
_REMOVE_BATCH_DIM_CASES = [
    ((), 0, 7),  # rank-0: batch becomes the only dim
    ((16,), 0, 7),  # rank-1, out_dim at the front
    ((16,), 1, 16),  # rank-1, out_dim at the end (batch == s0)
    ((1,), 0, 5),  # rank-1, size-1 dim at the front
    ((1,), 1, 9),  # rank-1, size-1 dim broadcast at the end
    ((256,), 0, 11),  # rank-1 regular 1-D shape, front
    ((256,), 1, 256),  # rank-1, out_dim at the end (batch == s0)
    ((64, 32), 0, 13),  # rank-2, out_dim at the front
    ((64, 32), 1, 64),  # rank-2, batch matches dim0
    ((1, 32), 1, 7),  # rank-2, size-1 broadcast at dim0
    ((1024, 1024), 0, 1),  # rank-2 regular 2-D shape, batch size 1
    ((2, 19, 7), 0, 5),  # rank-3, out_dim at the front
    ((2, 19, 7), 1, 2),  # rank-3, batch matches dim0
    ((1, 19, 7), 1, 5),  # rank-3, size-1 broadcast at dim0
    ((1, 19, 7), 2, 19),  # rank-3, middle out_dim, size-1 dim0
    ((4, 4, 16), 2, 4),  # rank-3, middle out_dim, adjacent dims equal
    ((4, 8, 16, 32), 0, 9),  # rank-4, out_dim at the front
    ((4, 8, 16, 32), 1, 4),  # rank-4, batch matches dim0
    ((8, 8, 8, 32), 2, 8),  # rank-4, middle out_dim
    ((1, 8, 16, 32), 2, 8),  # rank-4, middle out_dim, size-1 dim0
    ((16, 7, 57, 32, 29), 0, 1),  # rank-5, out_dim at the front
    ((1, 7, 57, 32, 29), 1, 11),  # rank-5, size-1 dim0 broadcast
]

# The op is a pure broadcast view: no arithmetic is performed, so every storage
# dtype is supported.
_REMOVE_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)

# Representative rank-3 (shape, out_dim, batch_size) used by the value-range
# sweep. The broadcast-view semantics do not depend on the values, so a single
# case is enough to verify that every value range in the spec passes through
# unchanged (int/bool bit-exact, floats with equal_nan=True).
_REMOVE_BATCH_DIM_VALUE_CASE = ((2, 19, 7), 1, 2)

# Backward of expand reduces the gradient over every broadcast (stride-0) dim,
# so the three cases below cover: a batch dim matching dim0 (sum over the new
# batch dim), a front batch dim, and a size-1 dim broadcast (sum over the
# expanded size-1 dim and the new batch dim).
_BACKWARD_CASES = [
    ((2, 19, 7), 1, 2),
    ((4, 8, 16, 32), 0, 9),
    ((1, 19, 7), 2, 19),
]


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
@pytest.mark.parametrize("shape, out_dim, batch_size", _REMOVE_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim(shape, out_dim, batch_size, level, dtype):
    # Values are irrelevant to the view itself (a representative [-1, 1] range
    # keeps every storage dtype valid); the dedicated value-range test below
    # sweeps the full spec ranges.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_value_ranges(value_range, dtype):
    # The broadcast view must repeat the exact stored values for every numeric
    # range in the spec (tu.assert_result_close compares int/bool bit-exactly
    # and floats with equal_nan=True). level is vmap bookkeeping: fixed at 0.
    shape, out_dim, batch_size = _REMOVE_BATCH_DIM_VALUE_CASE
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, 0, batch_size, out_dim)

    assert res_out.shape == ref_out.shape == _expected_shape(shape, out_dim, batch_size)
    assert res_out.dtype == ref_out.dtype == inp.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize(
    "shape, out_dim, batch_size",
    [((8, 16, 32), 1, 8), ((8, 8, 16, 32), 2, 8)],
)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_non_contiguous(shape, out_dim, batch_size, level, dtype):
    # The broadcast view must preserve the strides and storage offset of a
    # non-contiguous input. Slice on both the test device and the reference
    # device so the two inputs share the same memory layout. The sliced shape
    # is what out_dim/batch_size must be valid for.
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    _assert_output(res_out, ref_out, inp.shape, out_dim, batch_size, dtype)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__remove_batch_dim_nan_inf(dtype):
    # The view never performs arithmetic, so nan/inf/-inf and signed zeros pass
    # through the broadcast untouched (tu.assert_result_close compares floats
    # with equal_nan=True). 1e30 also covers the overflow-to-inf path in
    # fp16/bf16.
    vals = [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        1.5,
        -2.5,
        1e30,
        -1e30,
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device).reshape(3, 3)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, 4, 0)
    res_out = _resolve_gems_op()(inp, 0, 4, 0)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__remove_batch_dim_backward(case, dtype):
    # expand is differentiable: the gradient of the broadcast view is the
    # sum-reduction of grad_output over every broadcast (stride-0) dim. The
    # candidate gradient must match aten's, including the reduction.
    shape, out_dim, batch_size = case
    level = 0
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = utils.to_reference(inp)
    out_shape = _expected_shape(shape, out_dim, batch_size)
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_grad_out = utils.to_reference(grad_out)

    inp.requires_grad_(True)
    ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = _resolve_gems_op()(inp, level, batch_size, out_dim)

    res_grad = torch.autograd.grad(res_out, inp, grad_out)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)[0]

    assert res_grad.shape == ref_grad.shape == inp.shape
    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_non_broadcastable_batch_size():
    # Inserting batch_size=3 at out_dim=1 into (2, 19, 7) targets (2, 3, 19, 7):
    # dim 0 of self is 2 and can neither equal 3 nor broadcast from 1, so aten
    # raises RuntimeError and the candidate must too.
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, 3, 1)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 0, 3, 1)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_negative_batch_size():
    # A negative batch_size is rejected by expand; the candidate must reproduce
    # the validation.
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, -1, 0)
    with pytest.raises(RuntimeError):
        _resolve_gems_op()(inp, 0, -1, 0)
