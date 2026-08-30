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
from . import test_utils as tu  # noqa: E402

# aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor is the DDP
# gradient-flattening utility: it flattens every input to a contiguous 1-D
# tensor (t.contiguous().view(-1)) and concatenates them into one 1-D result.
# It is a pure data-movement op (copy + cat): inputs are never mutated, the
# result lives on the same device as the inputs, and every storage dtype aten
# supports round-trips bit-exactly (float incl. float64, int, and bool).
# nan/inf/+-0.0 pass through unchanged.
#
# Coverage follows the regular-operator spec adapted to a pure data-movement op
# whose shape dimension is a *list* of tensor shapes (there is no elementwise
# broadcast semantics to test):
#   * shape levels: single / several same-shape / mixed-rank / empty / 0-dim
#     input lists merged with the generic levels from tu.selected_shapes();
#   * value ranges: tu.selected_ranges() over small input lists for every
#     supported dtype (the values must round-trip exactly through the copy);
#   * edge cases: non-contiguous (transposed and strided) inputs and
#     nan/inf/-inf/+-0.0 passthrough;
#   * backward: autograd.grad() against the analytic narrow-and-view gradient
#     (grad_output[offset:offset+numel].view(input_shape));
#   * negative: an empty list and a non-tensor element raise on both paths.
_FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
_INT_DTYPES = utils.ALL_INT_DTYPES
_FLATTEN_DTYPES = _FLOAT_DTYPES + _INT_DTYPES + utils.BOOL_TYPES


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _flatten_shape_cases():
    """List-of-shapes workloads for the main shape-level sweep.

    Each case is a list of tensor shapes (the number, ranks, sizes and - via the
    dedicated non-contiguous test - the memory layout of the tensors fed to the
    op). Element counts stay <= 1M (except the single high-rank case, which is a
    pure copy and stays cheap) so correctness runs stay fast.
    """
    if tu.LEVEL == "quick":
        return [
            [(2, 19, 7)],
            [(2, 3), (4,), (5, 6, 7)],
        ]
    cases = [
        [(2, 3)],  # single tensor
        [(4, 5), (4, 5), (4, 5)],  # several same-shape tensors
        [(2, 3), (4,), (5, 6, 7)],  # mixed ranks and sizes
        [(1024,), (64, 64), (16, 16, 16)],  # larger tensors
        [(0, 3), (2,), (1, 1, 1)],  # empty tensor among non-empty
        [(), (3,), (1, 4)],  # 0-dim tensors
        [(0,), (0,)],  # all-empty tensors
        [(16, 7, 57, 32, 29)],  # high-rank single tensor
    ]
    for shape in tu.selected_shapes():
        numel = _numel(shape)
        if numel <= 1024 * 1024:
            cases.append([shape])
        if 2 * numel <= 1024 * 1024:
            cases.append([shape, shape])
    return cases


# Small input lists for the value-range sweep (the values are copied verbatim,
# so a few sizes suffice to exercise the full spec range list per dtype).
_FLATTEN_RANGE_CASES = [
    [(8,)],
    [(3,), (5,)],
    [(), (1,)],
    [(2, 4), (3, 3), (5,)],
]

# Small input lists for the backward sweep.
_FLATTEN_BACKWARD_CASES = [
    [(8,)],
    [(), (3,), (1, 4)],
    [(2, 3), (4,)],
    [(2, 4), (3, 3), (5,)],
    [(16, 16), (8, 8, 8)],
]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. Resolution order:
    # (1) override, (2) the direct flag_gems.flatten_dense_tensors callable,
    # (3) None when neither exists yet (the tests then fall back to the PyTorch
    # reference, keeping them runnable before an implementation is merged).
    try:
        return flag_gems.testing.resolve_gems_op(
            "flatten_dense_tensors", getattr(flag_gems, "flatten_dense_tensors", None)
        )
    except LookupError:
        return None


def _apply_flatten_dense_tensors(inp):
    gems_op = _resolve_gems_op()
    if gems_op is None:
        # No candidate injected and no native implementation registered yet:
        # run the reference so the test remains runnable standalone.
        return torch.ops.aten.flatten_dense_tensors(inp)
    return gems_op(inp)


def _assert_flattened(res_out, ref_out, dtype, input_device):
    # The result is a 1-D tensor of the input dtype on the input device holding
    # every input element in order (contiguous copy then concatenate).
    assert res_out.dim() == 1
    assert res_out.dtype == ref_out.dtype
    assert res_out.dtype == dtype
    assert res_out.numel() == ref_out.numel()
    assert res_out.device == input_device
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _flatten_shape_cases())
@pytest.mark.parametrize("dtype", _FLATTEN_DTYPES)
def test_flatten_dense_tensors(tensor_shapes, dtype):
    # Shape levels x every supported dtype, with values drawn from the default
    # [-1, 1] range (negative and positive for each dtype).
    inp = [tu.make_input(dtype, shape, ["-1", "1"]) for shape in tensor_shapes]
    inp_before = [t.clone() for t in inp]
    ref_inp = [utils.to_reference(t) for t in inp]
    expected_numel = sum(t.numel() for t in inp)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = _apply_flatten_dense_tensors(inp)

    _assert_flattened(res_out, ref_out, dtype, inp[0].device)
    assert res_out.numel() == expected_numel
    # The op is read-only: inputs must be left untouched.
    for res_t, before in zip(inp, inp_before):
        utils.gems_assert_equal(res_t, utils.to_reference(before))


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _FLATTEN_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES + _INT_DTYPES)
def test_flatten_dense_tensors_value_ranges(tensor_shapes, value_range, dtype):
    # The op never transforms the stored values, so the full spec range sweep
    # (including 0/max/min and the degenerate constant ranges) must round-trip
    # exactly through the copy. bool ignores the range and is covered by the
    # shape-level test above.
    inp = [tu.make_input(dtype, shape, value_range) for shape in tensor_shapes]
    ref_inp = [utils.to_reference(t) for t in inp]

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = _apply_flatten_dense_tensors(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", _FLATTEN_DTYPES)
def test_flatten_dense_tensors_non_contiguous(dtype):
    # Column slices and a transpose exercise the contiguous-copy path of the op.
    base = tu.make_input(dtype, (8, 16), ["-1", "1"])
    ref_base = utils.to_reference(base)
    views = [base[:, ::2], base.t(), base.reshape(4, 32)[:, ::3]]
    ref_views = [ref_base[:, ::2], ref_base.t(), ref_base.reshape(4, 32)[:, ::3]]
    assert all(not v.is_contiguous() for v in views)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_views)
    res_out = _apply_flatten_dense_tensors(views)

    _assert_flattened(res_out, ref_out, dtype, views[0].device)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_flatten_dense_tensors_nan_inf(dtype):
    # flatten_dense_tensors is a pure copy: +inf/-inf/nan/+-0.0 pass through
    # unchanged (equal_nan=True is active on the float path of
    # assert_result_close; 1e30 overflows to inf in fp16/bf16 on both paths
    # identically).
    values = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.5, -2.5, 1e30, -1e30],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor([1.0, -1.0], dtype=dtype, device=flag_gems.device)
    ref_inp = [utils.to_reference(values), utils.to_reference(other)]

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    res_out = _apply_flatten_dense_tensors([values, other])

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("tensor_shapes", _FLATTEN_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors_backward(tensor_shapes, dtype):
    # The forward op places input i at out[offset:offset+numel] after flattening
    # it to 1-D, so the gradient of input i is grad_output[offset:offset+numel]
    # viewed as the original shape (a pure narrow-and-view gather, no
    # arithmetic). Validate the autograd reference against that analytic value,
    # then check the candidate forward output and - only when the candidate
    # output is differentiable - its gradient against the reference gradient.
    inp = [
        tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
        for shape in tensor_shapes
    ]
    total_numel = sum(_numel(shape) for shape in tensor_shapes)
    grad = tu.make_input(dtype, (total_numel,), ["-1", "1"])
    ref_inp = [utils.to_reference(t) for t in inp]
    ref_grad = utils.to_reference(grad)

    ref_out = torch.ops.aten.flatten_dense_tensors(ref_inp)
    ref_in_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    expected = []
    offset = 0
    for shape in tensor_shapes:
        numel = _numel(shape)
        expected.append(ref_grad[offset : offset + numel].view(shape))
        offset += numel
    for got, exp in zip(ref_in_grads, expected):
        tu.assert_result_close(got, exp)

    res_out = _apply_flatten_dense_tensors(inp)
    tu.assert_result_close(res_out, ref_out)

    if res_out.requires_grad:
        res_in_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
        for got, exp in zip(res_in_grads, expected):
            tu.assert_result_close(got, exp)


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_empty_list():
    # aten requires a non-empty tensor list; the candidate must fail too.
    with pytest.raises(RuntimeError):
        torch.ops.aten.flatten_dense_tensors([])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, IndexError)):
            gems_op([])


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_non_tensor():
    # The tensors argument must be a list of Tensors; a scalar element hits a
    # schema mismatch and raises on both paths.
    a = tu.make_input(torch.float32, (4,), ["-1", "1"])
    ref_a = utils.to_reference(a)
    with pytest.raises(RuntimeError):
        torch.ops.aten.flatten_dense_tensors([ref_a, 3.14])
    gems_op = _resolve_gems_op()
    if gems_op is not None:
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            gems_op([a, 3.14])
