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
from . import test_utils as tu

# aten::coalesce(Tensor(a) self) -> Tensor(a) is the public method variant of
# sparse-tensor coalescing. It merges the duplicate entries of an uncoalesced
# sparse COO tensor: the result has unique, lexicographically sorted indices
# and each stored value is the sum of the entries sharing that index. Unlike
# aten::_coalesce (which asserts an uncoalesced input internally), coalesce
# also accepts an already-coalesced tensor and, per its ``Tensor(a)`` alias
# annotation, returns the input itself unchanged. Coalesce never mutates its
# input in the uncoalesced branch.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shapes: 1-D .. 5-D sparse COO layouts. The dense ``tu.selected_shapes()``
#     grid (which includes a 0-dim scalar) has no sparse COO analogue, so a
#     local sparse grid replaces it; ``nnz`` is always > numel(shape), which by
#     the pigeonhole principle guarantees duplicate coordinates and therefore
#     real merging work.
#   * value ranges: the spec's five ranges from ``tu.selected_ranges()`` fed
#     through ``tu.make_input``. Unsigned dtypes drop the negative ranges, and
#     ranges whose bound is the dtype extreme are skipped for narrow dtypes
#     because coalescing *sums* duplicates (see the accumulator-width note
#     below).
#   * identity branch: an already-coalesced input must be returned as itself.
#   * edge cases: nan / +-inf / -inf values for float dtypes.
#   * negative: dense, SparseCsr and float8 storage inputs have no registered
#     kernel and must raise.
#
# Broadcast does not apply (coalesce is unary, there is nothing to broadcast
# against). Sparse COO autograd has no formula for coalesce, so there is no
# backward test either.
#
# The CUDA implementation asserts ``!self.is_coalesced()`` internally, so every
# uncoalesced input here is produced by ``torch.sparse_coo_tensor`` with
# duplicate index rows (it does not set the coalesced flag). The tests assert
# that the candidate also leaves its input uncoalesced, i.e. it must not
# coalesce in place.

# Probe the storage dtypes the sparse COO coalesce kernel actually accepts on
# the active device (spec: never guess). fp8 raises
# ``"coalesce_sparse_cuda" not implemented for 'Float8_e4m3fn'`` on CUDA and is
# therefore excluded; every other required dtype is accepted.
_REQUIRED_CANDIDATE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


def _sparse_dtype_supported(dtype):
    # Three duplicate coordinates guarantee an uncoalesced input, so the probe
    # fails exactly when the dtype has no registered coalesce kernel.
    indices = torch.zeros((2, 3), dtype=torch.long, device=flag_gems.device)
    try:
        values = torch.zeros(3, dtype=dtype, device=flag_gems.device)
        inp = torch.sparse_coo_tensor(indices, values, (2, 2), device=flag_gems.device)
        torch.ops.aten.coalesce(inp)
    except Exception:
        return False
    return True


_COALESCE_DTYPES = [
    dtype for dtype in _REQUIRED_CANDIDATE_DTYPES if _sparse_dtype_supported(dtype)
]

# (shape, nnz) sparse layouts. nnz is always > numel(shape), which forces at
# least one duplicate coordinate and therefore real merging work. Ranks 1-5 and
# a mixture of tiny / medium / larger index spaces are covered.
_COALESCE_CASES = [
    ((8,), 20),
    ((4, 4), 20),
    ((5, 5), 30),
    ((8, 8), 80),
    ((16, 16), 300),
    ((2, 3, 4), 28),
    ((3, 5, 7), 120),
    ((4, 8, 16), 600),
    ((2, 3, 4, 5), 300),
    ((2, 3, 4, 5, 6), 1000),
]

# --quick smoke subset selected by tests/conftest.QUICK_MODE (tu.LEVEL).
_COALESCE_CASES_QUICK = [((4, 4), 20), ((3, 5, 7), 120)]

# The identity branch (input already coalesced -> returns self) is exercised on
# a subset of the shapes above; the input is made coalesced via .coalesce().
_COALESCED_CASES = [((5, 5), 30), ((3, 5, 7), 120), ((2, 3, 4, 5), 300)]
_COALESCED_CASES_QUICK = [((3, 5, 7), 120)]

# Representative sparse layouts for the value-range sweep (a 1-D, a 3-D and a
# 4-D index space), so every range is exercised on more than one rank.
_VALUE_RANGE_CASES = [((8,), 20), ((3, 5, 7), 120), ((4, 4, 4, 4), 400)]
_VALUE_RANGE_CASES_QUICK = [((3, 5, 7), 120)]

# Coalescing sums duplicate values, so ranges whose bound is the dtype's
# extreme are only swept for dtypes wide enough that the accumulated sum of a
# few duplicates cannot wrap / overflow differently depending on the
# accumulator width. Same-sign ranges avoid fp16/bf16 cancellation error.
_NARROW_DTYPES = (torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16)
_EXTREME_RANGES = (("0", "max"), ("min", "0"))
# Half-precision floats cannot even absorb the mixed-sign [-1, 1] range: two
# opposite-signed duplicates can cancel to a near-zero sum whose rounding error
# exceeds the absolute tolerance (most visible when the reference runs on the
# CPU, i.e. --ref cpu). Their sweep therefore stays same-sign.
_NARROW_FLOAT_DTYPES = (torch.float16, torch.bfloat16)


def _coalesce_cases():
    return _COALESCE_CASES_QUICK if tu.LEVEL == "quick" else _COALESCE_CASES


def _coalesced_cases():
    return _COALESCED_CASES_QUICK if tu.LEVEL == "quick" else _COALESCED_CASES


def _value_range_cases():
    cases = []
    shapes = _VALUE_RANGE_CASES_QUICK if tu.LEVEL == "quick" else _VALUE_RANGE_CASES
    for dtype in _COALESCE_DTYPES:
        if dtype == torch.bool:
            # A single degenerate {0, 1} range; the main grid covers bool.
            continue
        for value_range in tu.selected_ranges():
            if dtype == torch.uint8 and value_range in (["-1", "1"], ["-1", "0"]):
                # make_tensor cannot represent a negative bound for uint8.
                continue
            if dtype in _NARROW_FLOAT_DTYPES and value_range == ["-1", "1"]:
                continue
            if tuple(value_range) in _EXTREME_RANGES and dtype in _NARROW_DTYPES:
                continue
            for case in shapes:
                cases.append((value_range, dtype, case))
    return cases


def _default_bounds(dtype):
    if dtype.is_floating_point or dtype == torch.bool:
        return 0.0, 1.0
    info = torch.iinfo(dtype)
    return (-5.0, 6.0) if info.min < 0 else (0.0, 6.0)


def _make_input(shape, nnz, dtype, value_range=None, seed=2026):
    # Deterministic CPU-side index generation; the sparse tensor is created on
    # the test device. Index rows are drawn with replacement, so duplicates are
    # guaranteed whenever nnz > numel. Values come from the shared value-range
    # helper (tu.make_input) when a spec range is given; otherwise float values
    # default to [0, 1] (non-negative, so summing duplicates in a different
    # order cannot cancel) and signed integer dtypes use a small symmetric
    # range that cannot overflow.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    if value_range is not None:
        values = tu.make_input(dtype, (nnz,), value_range).cpu()
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    elif dtype.is_floating_point:
        low, high = _default_bounds(dtype)
        values = (
            torch.rand(nnz, dtype=torch.float64, generator=gen) * (high - low) + low
        ).to(dtype)
    else:
        low, high = _default_bounds(dtype)
        low_i, high_i = int(low), int(high)
        values = torch.randint(
            low_i, high_i, (nnz,), dtype=torch.int64, generator=gen
        ).to(dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_nan_inf_values(nnz, dtype):
    # Repeating pattern of nan / +inf / -inf / finite values. Coalescing maps an
    # index to nan when any contributor is nan (or when both +inf and -inf
    # contribute); the pattern is fully determined by the value multiset, so the
    # reference and the candidate agree with equal_nan=True.
    base = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.5,
            -0.5,
            2.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            1.5,
        ],
        dtype=dtype,
    )
    return base.repeat((nnz + base.numel() - 1) // base.numel())[:nnz]


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins.
    return flag_gems.testing.resolve_gems_op(
        "coalesce", getattr(flag_gems, "coalesce", None)
    )


def _assert_coalesced(res_out, ref_out, dtype):
    # Both sides must be coalesced sparse COO tensors with the same structure.
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced()
    assert ref_out.is_coalesced()
    # Indices are int64 and must match exactly (unique, sorted coordinates).
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    # Values are sums of duplicates: tolerance for float, exact for int/bool.
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.coalesce
@pytest.mark.parametrize("case", _coalesce_cases())
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test_coalesce(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # Coalescing returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark.coalesce
@pytest.mark.parametrize("case", _coalesced_cases())
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test_coalesce_coalesced_input(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype).coalesce()
    assert inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # Per the Tensor(a) alias annotation, coalesce returns the (coalesced)
    # input itself instead of a fresh tensor.
    assert ref_out is ref_inp
    assert res_out is inp


@pytest.mark.coalesce
@pytest.mark.parametrize("value_range,dtype,case", _value_range_cases())
def test_coalesce_value_ranges(value_range, dtype, case):
    # The stored values sweep the spec's five ranges via the shared
    # tu.make_input helper (positive, negative, extreme and degenerate); the
    # summed duplicate value must still match the reference within tolerance.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range=value_range)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark.coalesce
@pytest.mark.parametrize("case", _coalesce_cases())
@pytest.mark.parametrize(
    "dtype", [dtype for dtype in _COALESCE_DTYPES if dtype.is_floating_point]
)
def test_coalesce_nan_inf(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    # Rebuild with the same duplicate indices but values drawn from the
    # nan/inf/-inf pattern (the result pattern is deterministic).
    inp = torch.sparse_coo_tensor(
        inp._indices().clone(),
        _make_nan_inf_values(nnz, dtype),
        shape,
        device=flag_gems.device,
    )
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    # .indices() on an uncoalesced tensor raises, so this also proves the
    # structure is right.
    assert res_out.is_coalesced()
    assert ref_out.is_coalesced()
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype, equal_nan=True)
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark.coalesce
def test_coalesce_rejects_dense_input():
    # coalesce is a sparse-COO-only operator; a dense (strided) input has no
    # registered kernel and must raise. NotImplementedError is a RuntimeError
    # subclass, so the candidate is held to the same contract on any device.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.coalesce
def test_coalesce_rejects_csr_input():
    # Sparse CSR layout is not COO: coalesce must reject it as well.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten.coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark.coalesce
def test_coalesce_rejects_fp8_input():
    # float8 storage has no registered coalesce kernel; the candidate must
    # reject it too rather than silently producing a bogus result.
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = torch.zeros(3, dtype=torch.float8_e4m3fn, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)
