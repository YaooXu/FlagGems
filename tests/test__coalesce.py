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
from . import test_utils as tu

# ``_coalesce`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._coalesce``,
# ``@pytest.mark._coalesce_out`` and ``-m _coalesce`` all work.
setattr(
    pytest.mark,
    "_coalesce",
    MarkDecorator(Mark("_coalesce", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_coalesce_out",
    MarkDecorator(Mark("_coalesce_out", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_coalesce(Tensor self) -> Tensor merges the duplicate entries of an
# *uncoalesced* sparse COO tensor: the result has unique, lexicographically
# sorted coordinates and each stored value is the sum of the entries sharing
# that coordinate. aten::_coalesce.out(Tensor self, *, Tensor(a!) out) writes
# the same result into a caller-provided (empty) sparse COO tensor.
#
# Coverage (regular-operator spec, sparse/metadata adaptation):
#   * shapes: 1-D .. 4-D sparse COO layouts (local set -- the shared
#     ``tu.selected_shapes()`` set is dense-only and has no sparse analogue),
#     always with nnz > numel so the pigeonhole principle guarantees duplicate
#     entries and coalescing has real merging work to do;
#   * value ranges: ``tu.selected_ranges()`` (the spec's five ranges) fed
#     through ``tu.make_input``; unsupported negative bounds for uint8 and
#     overflow-prone extremes for narrow dtypes are filtered (coalescing
#     *sums* duplicates, see below);
#   * .out overload: the same grid, with an empty out buffer and return/alias
#     checks;
#   * edge cases: nan / +-inf values (float dtypes);
#   * negative: dense, SparseCsr and float8 storage inputs have no registered
#     kernel and must raise.
#
# Broadcast and backward do not apply: ``_coalesce`` is unary (nothing to
# broadcast against) and sparse COO autograd has no formula for it (the
# operator only re-orders/merges stored values, there is no gradient rule to
# exercise).
#
# The CUDA implementation asserts ``!self.is_coalesced()`` internally, so every
# input here is left uncoalesced (``torch.sparse_coo_tensor`` does not set the
# coalesced flag on this runtime); the tests assert that the *candidate* also
# leaves its input uncoalesced, i.e. it must not coalesce in place.

# Probe the storage dtypes the sparse COO ``_coalesce`` kernel actually accepts
# on the active device (spec: never guess). fp8 raises
# ``"coalesce_sparse_cuda" not implemented for 'Float8_e4m3fn'`` on CUDA and is
# therefore excluded; every other required dtype is accepted. complex64 is
# accepted by the CUDA kernel as well but is outside the spec's required dtype
# contract, so it is not swept.
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
        torch.ops.aten._coalesce(inp)
    except Exception:
        return False
    return True


_COALESCE_DTYPES = [
    dtype for dtype in _REQUIRED_CANDIDATE_DTYPES if _sparse_dtype_supported(dtype)
]

# (shape, nnz) sparse layouts. nnz is always > numel(shape), which forces at
# least one duplicate coordinate and therefore real merging work. Ranks 1-4
# and a mixture of tiny / medium / larger index spaces are covered.
_COALESCE_CASES = [
    ((4, 4), 20),
    ((5, 5), 30),
    ((8, 8), 80),
    ((16, 16), 300),
    ((64,), 200),
    ((2, 3, 4), 28),
    ((3, 5, 7), 120),
    ((4, 8, 16), 600),
    ((16, 7, 57), 2000),
    ((4, 4, 4, 4), 400),
]

# --quick smoke subset selected by tests/conftest.QUICK_MODE (tu.LEVEL).
_COALESCE_CASES_QUICK = [((4, 4), 20), ((3, 5, 7), 120)]

# Representative layouts for the value-range sweep (a 1-D, a 3-D and a 4-D
# index space), so every range is exercised on more than one rank.
_VALUE_RANGE_CASES = [((64,), 200), ((3, 5, 7), 120), ((4, 4, 4, 4), 400)]
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


def _make_input(shape, nnz, dtype, value_range=None, low=None, high=None, seed=2026):
    # Deterministic CPU-side index generation; the sparse tensor is created on
    # the test device. Index rows are drawn with replacement, so duplicates are
    # guaranteed whenever nnz > numel. Values come from the shared value-range
    # helper (tu.make_input) when a spec range is given; otherwise duplicate
    # values are summed by coalescing, so the default float range is
    # non-negative ([0, 1]) to avoid fp16/bf16 cancellation error and signed
    # integer dtypes default to a small symmetric range that cannot overflow.
    gen = torch.Generator("cpu").manual_seed(seed)
    if low is None or high is None:
        low, high = _default_bounds(dtype)
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
        # Generate in float64 so that dtype-extreme bounds (e.g. finfo.min/max)
        # are representable while sampling; casting afterwards keeps extreme
        # float64 values from turning into inf/nan during the draw.
        values = (
            torch.rand(nnz, dtype=torch.float64, generator=gen)
            * (float(high) - float(low))
            + float(low)
        ).to(dtype)
    else:
        low_i, high_i = int(low), int(high)
        if high_i <= low_i:
            values = torch.full((nnz,), low_i, dtype=torch.int64, device="cpu").to(
                dtype
            )
        else:
            values = torch.randint(
                low_i, high_i, (nnz,), dtype=torch.int64, generator=gen
            ).to(dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_nan_inf_values(nnz, dtype):
    # Repeating pattern of nan / +inf / -inf / finite values. Coalescing maps a
    # coordinate to nan when any contributor is nan (or when both +inf and -inf
    # contribute); the pattern is fully determined by the value multiset, so
    # the reference and the candidate agree with equal_nan=True.
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


def _make_empty_out(shape, dtype, device):
    # Empty sparse COO tensor of the right shape/dtype; _coalesce.out writes the
    # coalesced indices and values into this storage.
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # .default and .out overloads are resolved through their public operator
    # names "_coalesce" and "_coalesce.out".
    return flag_gems.testing.resolve_gems_op(
        "_coalesce", getattr(flag_gems, "_coalesce", None)
    )


def _resolve_gems_op_out():
    return flag_gems.testing.resolve_gems_op(
        "_coalesce.out", getattr(flag_gems, "_coalesce_out", None)
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


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _coalesce_cases())
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # Coalescing returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("value_range,dtype,case", _value_range_cases())
def test__coalesce_value_ranges(value_range, dtype, case):
    # The stored values sweep the spec's five ranges via the shared
    # tu.make_input helper (positive, negative, extreme and degenerate); the
    # summed duplicate value must still match the reference within the dtype
    # tolerance.
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range=value_range)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce
@pytest.mark.parametrize("case", _coalesce_cases())
@pytest.mark.parametrize(
    "dtype", [dtype for dtype in _COALESCE_DTYPES if dtype.is_floating_point]
)
def test__coalesce_nan_inf(case, dtype):
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

    ref_out = torch.ops.aten._coalesce(ref_inp)
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


@pytest.mark._coalesce_out
@pytest.mark.parametrize("case", _coalesce_cases())
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce_out(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())
    out = _make_empty_out(shape, dtype, flag_gems.device)
    ref_out = _make_empty_out(shape, dtype, ref_inp.device)

    ref_ret = torch.ops.aten._coalesce.out(ref_inp, out=ref_out)
    res_ret = _resolve_gems_op_out()(inp, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    assert ref_ret is ref_out
    _assert_coalesced(res_ret, ref_ret, dtype)
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark._coalesce
def test__coalesce_rejects_dense_input():
    # _coalesce is a sparse-COO-only operator; a dense (strided) input has no
    # registered kernel and must raise. NotImplementedError is a RuntimeError
    # subclass, so the candidate is held to the same contract on any device.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._coalesce
def test__coalesce_rejects_csr_input():
    # Sparse CSR layout is not COO: _coalesce must reject it as well.
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _resolve_gems_op()(inp)


@pytest.mark._coalesce
def test__coalesce_rejects_fp8_input():
    # float8 storage has no registered coalesce kernel; the candidate must
    # reject it too rather than silently producing a bogus result.
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = torch.zeros(3, dtype=torch.float8_e4m3fn, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
