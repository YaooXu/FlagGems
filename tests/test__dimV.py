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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimV(Tensor self) -> int reports the number of dense dimensions of a
# sparse tensor (``dense_dim``): a pure metadata query whose result never
# depends on the stored index/data values or on the storage dtype.
#
# The SparseCPU / SparseCUDA / SparseMeta / SparseXPU backends are the only
# dispatch targets (dense and SparseCsr* tensors raise NotImplementedError), so
# every workload below feeds a sparse COO tensor.
#
# Coverage (regular-operator spec):
#   * value ranges: the five shared ranges ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) swept over the shared shape levels and a set of hybrid layouts,
#     so every storage dtype sees negative, positive, extreme and degenerate
#     inputs;
#   * shapes: the shared tu.selected_shapes() levels (rank >= 1; a rank-0 sparse
#     tensor has no dimension to report) paired with a small hybrid dense tail,
#     plus explicit all-sparse/hybrid higher-rank layouts selected by the pytest
#     --quick flag;
#   * dtypes: int8 / uint8 / float8_e4m3fn / float8_e5m2 / fp32 / bf16 / fp16 /
#     int32 / int64 (the spec's required set) plus fp64/int16/bool, filtered by
#     a device probe so backends that cannot store a dtype are skipped cleanly;
#   * boundary cases: empty (nnz == 0, all-sparse and hybrid), single entry,
#     uncoalesced, nan/inf/-inf/±0.0 payloads (all ignored by the query);
#   * negative cases: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a plain
# Python int, and has no autograd formula (there is nothing to broadcast against
# or differentiate).

# Required dtype coverage first, then the shared float/int/bool sets (the probe
# below removes duplicates and anything the active backend cannot build).
_DIMV_DTYPE_CANDIDATES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + list(utils.ALL_FLOAT_DTYPES)
    + list(utils.ALL_INT_DTYPES)
    + list(utils.BOOL_TYPES)
)


def _make_values(dtype, shape, value_range):
    """Value-range helper that survives the unsigned-dtype snapping of the
    shared helper (uint8 cannot represent the ``-1`` / ``min`` low bound, which
    ``torch.testing.make_tensor`` rejects for a non-degenerate range)."""
    low, high = value_range
    dtype_min, _ = tu.dtype_bounds(dtype)
    if dtype_min >= 0 and low in ("-1", "min"):
        # Unsigned dtype: clamp the negative low bound to the representable set.
        low = "0"
    try:
        return tu.make_input(dtype, shape, [low, high])
    except RuntimeError:
        # Any other unrepresentable combination: fall back to the full
        # non-negative range instead of failing input generation.
        return tu.make_input(dtype, shape, ["0", "max"])


def _make_coo_input(shape, dense_dim, dtype, value_range, nnz=8, seed=0):
    """Build a sparse COO tensor whose reported ``dense_dim`` is ``dense_dim``.

    Deterministic CPU-side index generation; the values tensor comes from the
    shared value-range helper and the sparse tensor is created on the test
    device. Duplicate indices are allowed and merely leave the tensor
    uncoalesced (covered explicitly below).
    """
    shape = tuple(shape)
    if not 0 <= dense_dim < len(shape):
        raise ValueError("dense_dim must leave at least one sparse dimension")
    sparse_dim = len(shape) - dense_dim
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + tuple(dense_shape), value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_empty_coo(shape, dense_dim, dtype):
    shape = tuple(shape)
    sparse_dim = len(shape) - dense_dim
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(
        (0,) + shape[sparse_dim:], dtype=dtype, device=flag_gems.device
    )
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _supported_sparse_dtypes():
    """Probe which candidate dtypes can be stored in a sparse COO tensor that
    ``_dimV`` accepts on the active device.

    ``tu.supported_dtypes`` builds a *dense* probe input, which always raises
    NotImplementedError for this op, so the check has to go through a sparse
    tensor. Any exception (missing sparse/fp8 storage support or a missing op
    kernel) marks the dtype unsupported. Falls back to the shared float/int/bool
    sets if the probe cannot establish anything, so the file never collects zero
    cases.
    """
    supported = []
    for dtype in _DIMV_DTYPE_CANDIDATES:
        if dtype in supported:
            continue
        try:
            values = _make_values(dtype, (3, 2), ["0", "1"])
            indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
            inp = torch.sparse_coo_tensor(
                indices, values, (4, 2), device=flag_gems.device
            )
            ref = torch.ops.aten._dimV(inp)
        except Exception:
            continue
        if isinstance(ref, int) and not isinstance(ref, bool):
            supported.append(dtype)
    if not supported:
        return (
            list(utils.ALL_FLOAT_DTYPES)
            + list(utils.ALL_INT_DTYPES)
            + list(utils.BOOL_TYPES)
        )
    return supported


_DIMV_DTYPES = _supported_sparse_dtypes()
_DIMV_FLOAT_DTYPES = [dtype for dtype in _DIMV_DTYPES if dtype.is_floating_point]

# (shape, dense_dim) pairs: dense_dim is the reported result and
# sparse_dim == len(shape) - dense_dim. Covers all-sparse (dense_dim == 0) and
# hybrid layouts over ranks 1-5.
_DIMV_COO_CASES_CORE = [
    ((5,), 0),
    ((3, 4), 0),
    ((3, 4), 1),
    ((8, 8, 8), 0),
    ((3, 4, 2), 1),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 3),
    ((3, 4, 5, 4, 5), 2),
]

# Higher-rank layouts for the "all" level (no --quick): all-sparse ranks up to
# 6-D and hybrid ranks up to 7-D.
_DIMV_COO_CASES_ALL = [
    ((12, 9, 3, 6), 0),
    ((3, 6, 4, 4, 6, 5), 2),
    ((7, 3, 12, 4, 2, 15), 3),
    ((3, 4, 2, 5, 3, 4, 2), 4),
    ((2, 4, 2, 4, 2, 4), 3),
]

# Small layouts kept in --quick mode: all-sparse plus two hybrid ranks, so the
# smoke level still exercises dense_dim == 0 and dense_dim > 0.
_DIMV_COO_CASES_QUICK = [
    ((2, 19, 7), 0),
    ((2, 19, 7), 1),
    ((2, 19, 7), 2),
    ((2, 19, 7, 5), 0),
    ((2, 19, 7, 5), 1),
    ((2, 19, 7, 5), 3),
    ((2, 19, 7, 5, 3), 0),
    ((2, 19, 7, 5, 3), 2),
]

# Representative hybrid layouts for the per-range sweep (dense_dim > 0).
_DIMV_HYBRID_CORE = [
    ((3, 4), 1),
    ((3, 4, 2), 1),
    ((4, 3, 4, 5), 3),
    ((3, 4, 5, 4, 5), 2),
]

# Dense-dim split used for the shared shape levels. The dense suffix is chosen
# small so the hybrid payload stays tiny even for the largest shape.
_SHAPE_DENSE_DIM = {
    (1,): 0,
    (256,): 0,
    (1024, 1024): 1,
    (20, 320, 15): 1,
    (16, 128, 64, 60): 2,
    (16, 7, 57, 32, 29): 3,
}


def _coo_cases():
    """(shape, dense_dim) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return _DIMV_COO_CASES_QUICK
    if tu.LEVEL == "all":
        return _DIMV_COO_CASES_CORE + _DIMV_COO_CASES_ALL


def _hybrid_value_range_cases():
    """Representative hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [
            ((2, 19, 7), 1),
            ((2, 19, 7), 2),
            ((2, 19, 7, 5), 1),
            ((2, 19, 7, 5, 3), 1),
        ]
    if tu.LEVEL == "all":
        return _DIMV_HYBRID_CORE


def _shape_level_cases():
    """The shared shape levels paired with a (small) hybrid dense tail.

    A rank-0 sparse tensor has no dimension to report, so only ranks >= 1 are
    kept; every other shared shape keeps its trailing dimension(s) dense.
    """
    cases = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        if len(shape) < 1:
            continue
        cases.append((shape, _SHAPE_DENSE_DIM.get(shape, 0)))
    return cases


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._dimV is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimV callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_dimV", getattr(flag_gems, "_dimV", None))


def _assert_result(res_out, ref_out, dense_dim):
    # _dimV returns a plain Python int holding the dense dimension count, so
    # exact equality is required and no tolerance is involved.
    assert isinstance(res_out, int) and not isinstance(res_out, bool)
    assert isinstance(ref_out, int) and not isinstance(ref_out, bool)
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == dense_dim


@pytest.mark._dimV
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values for
    # every storage dtype (bool/int snap the range to the representable set,
    # unsigned dtypes clamp the low bound). The reported count must be the
    # layout's dense dim.
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() == len(shape) - dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimV
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_shape_value_range_grid(case, value_range, dtype):
    # The full spec grid: the shared shape levels (rank 1-5) x the five required
    # value ranges x every supported dtype. The reported count is the dense dim
    # chosen for the layout; the payload never changes it.
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimV
@pytest.mark.parametrize("case", _hybrid_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_hybrid_value_ranges(case, value_range, dtype):
    # The five required value ranges on hybrid (sparse + dense) layouts: the
    # payload never changes the reported dense dim, only the layout does.
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)
    assert inp.dense_dim() > 0
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty(dtype):
    # nnz == 0: indices and values are empty, but the dense dims of the layout
    # are still reported exactly as for a populated tensor.
    shape, dense_dim = (3, 4), 0
    inp = _make_empty_coo(shape, dense_dim, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("shape, dense_dim", [((4, 5, 6), 1), ((4, 5, 6), 2)])
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty_hybrid(shape, dense_dim, dtype):
    # nnz == 0 with dense dimensions: the hybrid layout is preserved and the
    # dense dims stay exactly as for a populated tensor.
    inp = _make_empty_coo(shape, dense_dim, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_single_entry(dtype):
    # nnz == 1 boundary: a hybrid layout with a single stored entry still
    # reports the full dense dim count.
    shape, dense_dim = (3, 4, 5), 2
    inp = _make_coo_input(shape, dense_dim, dtype, ["-1", "1"], nnz=1)
    assert inp._nnz() == 1
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _dimV must still report
    # the same dense dim as the coalesced form (it never inspects the index or
    # data values). The (0, 0) coordinate is repeated three times.
    shape, dense_dim = (3, 4), 1
    indices = torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.long)
    values = _make_values(dtype, (5, 4), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, dense_dim)


@pytest.mark._dimV
@pytest.mark.parametrize("dtype", _DIMV_FLOAT_DTYPES)
def test__dimV_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: the metadata query still
    # reports the dense dim of the layout, independent of the payload.
    values = torch.tensor(
        [
            [float("nan"), float("inf")],
            [float("inf"), float("-inf")],
            [0.0, -0.0],
            [1.5, 2.5],
            [float("nan"), 1.0],
            [float("-inf"), 0.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6, 2), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, 1)


# A candidate may legitimately surface the "no sparse layout" failure as a
# Python-level error of any of these kinds, so all of them are accepted.
_NEGATIVE_EXC = (
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)


@pytest.mark._dimV
def test__dimV_dense_raises():
    # _dimV dispatches only on the sparse COO backends; dense tensors have no
    # implementation and raise. The candidate must fail too rather than silently
    # report a bogus count.
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(utils.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(inp)


@pytest.mark._dimV
def test__dimV_csr_raises():
    # SparseCsr* backends have no kernel for _dimV (unlike _nnz, which does
    # dispatch there): a CSR tensor raises instead of reporting its 0 dense
    # dims, and the candidate must fail too.
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = _make_values(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(utils.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(inp)


@pytest.mark._dimV
def test__dimV_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises. The candidate must reject it
    # too.
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimV(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(3.14)
