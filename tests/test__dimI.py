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

# ``_dimI`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimI`` and ``-m _dimI`` both work.
setattr(
    pytest.mark,
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimI(Tensor self) -> int reports the number of sparse dimensions of a
# sparse tensor (``sparse_dim``): a pure metadata query whose result never
# depends on the stored index/data values or on the storage dtype.
#
# The SparseCPU / SparseCUDA / SparseXPU backends are the only dispatch targets
# (dense and SparseCsr* tensors raise NotImplementedError), so every workload
# below feeds a sparse COO tensor.
#
# Coverage (regular-operator spec):
#   * value ranges: the five shared ranges ([-1,1], [0,1], [-1,0], [0,max],
#     [min,0]) swept over the shared shape levels, so every storage dtype sees
#     negative, positive, extreme and degenerate inputs;
#   * shapes: the shared tu.selected_shapes() levels mapped onto all-sparse COO
#     layouts (rank >= 1; a rank-0 sparse tensor does not exist), plus explicit
#     higher-rank and hybrid layouts selected by the pytest --quick flag;
#   * dtypes: int8 / uint8 / float8_e4m3fn / float8_e5m2 / fp32 / bf16 / fp16 /
#     int32 / int64 (the spec's required set) plus fp64/int16/bool, filtered by
#     a device probe so backends that cannot store a dtype are skipped cleanly;
#   * boundary cases: empty (nnz == 0, dense and hybrid), single entry,
#     uncoalesced, nan/inf/-inf/±0.0 payloads (all ignored by the query);
#   * negative cases: dense tensors, SparseCsr tensors and non-tensor inputs are
#     rejected.
#
# No broadcast/backward dimensions apply: the operator is unary, returns a plain
# Python int, and has no autograd formula (there is nothing to broadcast against
# or differentiate).

# Required dtype coverage first, then the shared float/int/bool sets (the probe
# below removes duplicates and anything the active backend cannot build).
_DIMI_DTYPE_CANDIDATES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + list(utils.ALL_FLOAT_DTYPES)
    + list(utils.ALL_INT_DTYPES)
    + list(utils.BOOL_TYPES)
)


def _supported_sparse_dtypes():
    """Probe which candidate dtypes can be stored in a sparse COO tensor that
    ``_dimI`` accepts on the active device.

    ``tu.supported_dtypes`` builds a *dense* probe input, which always raises
    NotImplementedError for this op, so the check has to go through a sparse
    tensor. Any exception (missing sparse/fp8 storage support or a missing op
    kernel) marks the dtype unsupported. Falls back to the shared float/int/bool
    sets if the probe cannot establish anything, so the file never collects zero
    cases.
    """
    supported = []
    for dtype in _DIMI_DTYPE_CANDIDATES:
        if dtype in supported:
            continue
        try:
            values = tu.make_input(dtype, (3,), ["0", "1"])
            indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
            inp = torch.sparse_coo_tensor(
                indices, values, (4,), device=flag_gems.device
            )
            ref = torch.ops.aten._dimI(inp)
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


_DIMI_DTYPES = _supported_sparse_dtypes()
_DIMI_FLOAT_DTYPES = [dtype for dtype in _DIMI_DTYPES if dtype.is_floating_point]

# (shape, sparse_dim) pairs covering 1-D/2-D/3-D all-sparse, hybrid layouts and
# mixed sparse+dense ranks up to 6-D.
_DIMI_COO_CASES_CORE = [
    ((5,), 1),
    ((3, 4), 2),
    ((3, 4), 1),
    ((8, 8, 8), 3),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 1),
    ((3, 4, 5, 4, 5), 3),
]

# Higher-rank layouts for the "all" level (no --quick): 4-D/5-D all-sparse and
# hybrid ranks up to 7-D.
_DIMI_COO_CASES_ALL = [
    ((12, 9, 3, 6), 4),
    ((3, 6, 4, 4, 6, 5), 4),
    ((7, 3, 12, 4, 2, 15), 5),
    ((3, 4, 2, 5, 3, 4, 2), 3),
    ((2, 4, 2, 4, 2, 4), 2),
]

# Small layouts kept in --quick mode: one all-sparse and two hybrid ranks, so
# the smoke level still exercises sparse_dim < ndim and sparse_dim == ndim.
_DIMI_COO_CASES_QUICK = [
    ((2, 19, 7), 2),
    ((2, 19, 7), 3),
    ((2, 19, 7, 5), 2),
]

# Representative hybrid layouts for the per-range sweep (sparse_dim < ndim).
_DIMI_HYBRID_CORE = [
    ((3, 4), 1),
    ((3, 4, 2), 2),
    ((4, 3, 4, 5), 1),
    ((3, 4, 5, 4, 5), 3),
]


def _coo_cases():
    """(shape, sparse_dim) layouts selected by pytest --quick (quick) vs default (full)."""
    if tu.LEVEL == "quick":
        return _DIMI_COO_CASES_QUICK
    if tu.LEVEL == "all":
        return _DIMI_COO_CASES_CORE + _DIMI_COO_CASES_ALL


def _hybrid_value_range_cases():
    """Representative hybrid layouts for the value-range sweep."""
    if tu.LEVEL == "quick":
        return [((2, 19, 7), 2)]
    if tu.LEVEL == "all":
        return _DIMI_HYBRID_CORE


def _shape_level_cases():
    """The shared shape levels mapped onto all-sparse COO layouts.

    A rank-0 sparse tensor does not exist, so only ranks >= 1 are kept; the
    remaining shared shapes use ``sparse_dim == ndim`` (all-sparse).
    """
    return [shape for shape in tu.selected_shapes() if len(shape) >= 1]


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


def _make_coo_input(shape, sparse_dim, dtype, value_range, nnz=8, seed=0):
    # Deterministic CPU-side index generation; the values tensor comes from the
    # shared value-range helper and the sparse tensor is created on the test
    # device. Duplicate indices are allowed and merely leave the tensor
    # uncoalesced (covered explicitly below).
    shape = tuple(shape)
    if sparse_dim < 1:
        raise ValueError("sparse COO tensors need at least one sparse dimension")
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + tuple(dense_shape), value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_empty_coo(shape, sparse_dim, dtype):
    shape = tuple(shape)
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(
        (0,) + shape[sparse_dim:], dtype=dtype, device=flag_gems.device
    )
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _resolve_gems_op():
    # Resolved inside each test (never at module import time) so the
    # process-local override injected by KernelGen for this run wins. The
    # default stays None until flag_gems._dimI is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._dimI callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op("_dimI", getattr(flag_gems, "_dimI", None))


def _assert_result(res_out, ref_out, sparse_dim):
    # _dimI returns a plain Python int holding the sparse dimension count, so
    # exact equality is required and no tolerance is involved.
    assert isinstance(res_out, int) and not isinstance(res_out, bool)
    assert isinstance(ref_out, int) and not isinstance(ref_out, bool)
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out == sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("case", _coo_cases())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_coo(case, dtype):
    # Layout coverage with values from [-1, 1]: negative and positive values for
    # every storage dtype (bool/int snap the range to the representable set,
    # unsigned dtypes clamp the low bound). The reported count must be the
    # layout's sparse dim.
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == sparse_dim
    assert inp.dense_dim() == len(shape) - sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("shape", _shape_level_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_shape_value_range_grid(shape, value_range, dtype):
    # The full spec grid on all-sparse layouts: the shared shape levels (1-5
    # dims) x the five required value ranges x every supported dtype. The
    # reported count is exactly len(shape) because dim == sparse_dim here.
    inp = _make_coo_input(shape, len(shape), dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, len(shape))
    assert inp.dense_dim() == 0


@pytest.mark._dimI
@pytest.mark.parametrize("case", _hybrid_value_range_cases())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_hybrid_value_ranges(case, value_range, dtype):
    # The five required value ranges on hybrid (sparse + dense) layouts: the
    # payload never changes the reported sparse dim, only the layout does.
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimI
@pytest.mark.parametrize("case", _DIMI_COO_CASES_CORE)
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_hybrid_dense_dim_zero(case, dtype):
    # Boundary sweep on the dense-dim side: for every core layout,
    # sparse_dim + dense_dim == ndim must hold both for the input and for the
    # reported count.
    shape, sparse_dim = case
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"])
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty(dtype):
    # nnz == 0: indices and values are empty, but the sparse dims of the layout
    # are still reported exactly as for a populated tensor.
    shape, sparse_dim = (3, 4), 2
    inp = _make_empty_coo(shape, sparse_dim, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("shape, sparse_dim", [((4, 5, 6), 2), ((4, 5, 6), 1)])
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_empty_hybrid(shape, sparse_dim, dtype):
    # nnz == 0 with dense dimensions: the hybrid layout is preserved and the
    # sparse dims stay exactly as for a populated tensor.
    inp = _make_empty_coo(shape, sparse_dim, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)
    assert inp.dense_dim() == len(shape) - sparse_dim


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_single_entry(dtype):
    # nnz == 1 boundary: a hybrid layout with a single stored entry still
    # reports the full sparse dim count.
    shape, sparse_dim = (3, 4, 5), 2
    inp = _make_coo_input(shape, sparse_dim, dtype, ["-1", "1"], nnz=1)
    assert inp._nnz() == 1
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_DTYPES)
def test__dimI_uncoalesced(dtype):
    # Duplicate indices leave the tensor uncoalesced; _dimI must still report
    # the same sparse dim as the coalesced form (it never inspects the index or
    # data values). The (0, 1) coordinate is repeated three times.
    shape, sparse_dim = (3, 4), 2
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_result(res_out, ref_out, sparse_dim)


@pytest.mark._dimI
@pytest.mark.parametrize("dtype", _DIMI_FLOAT_DTYPES)
def test__dimI_nan_inf_values_ignored(dtype):
    # nan/inf/-inf/±0.0 are ordinary stored values: the metadata query still
    # reports the sparse dim of the layout, independent of the payload.
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._dimI(ref_inp)
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


@pytest.mark._dimI
def test__dimI_dense_raises():
    # _dimI dispatches only on the sparse COO backends; dense tensors have no
    # implementation and raise. The candidate must fail too rather than silently
    # report a bogus count.
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(utils.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(inp)


@pytest.mark._dimI
def test__dimI_csr_raises():
    # SparseCsr* backends have no kernel for _dimI (unlike _nnz, which does
    # dispatch there): a CSR tensor raises instead of reporting its 2 sparse
    # dims, and the candidate must fail too.
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = _make_values(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimI(utils.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(inp)


@pytest.mark._dimI
def test__dimI_rejects_non_tensor():
    # The aten schema requires a Tensor; a Python scalar hits the invalid
    # combination of arguments path and raises. The candidate must reject it
    # too.
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimI(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        _resolve_gems_op()(3.14)
