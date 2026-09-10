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
from . import conftest as cfg
from . import test_utils as tu

# aten::sparse_coo_tensor overload group
# =====================================
# Three schemas share the public name ``sparse_coo_tensor`` and dispatch on the
# argument count / shape:
#
#   * ``sparse_coo_tensor(int[] size, *, dtype, layout, device, pin_memory)``
#     -- the size-only overload: an empty (nnz == 0) sparse tensor;
#   * ``sparse_coo_tensor(Tensor indices, Tensor values, *, ...)``
#     -- the size-inferred overload: the sparse part of the size is
#        ``max(index[d]) + 1`` per sparse dim and the dense part comes from the
#        values shape;
#   * ``sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ...)``
#     -- the explicit ``indices_size`` overload.
#
# A fourth schema, ``sparse_coo_tensor.size_out(int[] size, *, Tensor(a!) out)``,
# is a real, separately dispatched overload (verified callable on the active
# device): it writes the empty tensor into a caller-provided sparse COO buffer
# whose logical shape already matches and returns that same buffer. It is
# resolved through its own public name ``sparse_coo_tensor.size_out`` (default
# callable ``flag_gems.sparse_coo_tensor_size_out``).
#
# The candidate is the same public callable for the first three schemas --
# resolved inside every test through ``flag_gems.testing.resolve_gems_op`` (never
# at module import time, never through the dispatcher) -- and every reference
# call mirrors the candidate call exactly with
# ``torch.ops.aten.sparse_coo_tensor``. ``dtype`` is always passed explicitly
# (otherwise the aten op forces float32) and ``device`` is passed explicitly so
# the reference device (``--ref cpu`` or the active device) never has to be
# inferred from the component tensors.
#
# COO layout facts asserted below: ``layout == torch.sparse_coo``, indices shape
# ``(sparse_dim, nnz)`` with dtype int64, values shape
# ``(nnz,) + size[sparse_dim:]`` and ``sparse_dim + dense_dim == ndim``. The
# constructor stores the raw components verbatim (duplicate / unsorted
# coordinates stay uncoalesced, ``nnz == 0`` inputs are coalesced), so the
# candidate must reproduce the coalesced flag instead of normalising it.
#
# Regular-operator spec adaptation (sparse / metadata operator):
#   * value ranges -- the five spec ranges feed ``tu.make_input`` for every
#     supported dtype (float and exact), one pytest case per
#     (range, layout, dtype) combination; ranges a dtype cannot represent are
#     filtered at collection time (uint8 has no [-1, 0] range);
#   * shapes -- sparse COO indexing has no sensible 0-dim / 5-dim analogue and
#     the shared dense ``tu.selected_shapes()`` set does not map onto
#     (indices, values), so a dedicated sparse grid replaces it: 1..4 logical
#     dims, dense dims, nnz == 0 and a zero-extent logical dim;
#   * dtypes -- the full spec dtype contract probed locally (all nine required
#     dtypes plus float64 / int16 / bool are accepted by this factory on the
#     active device), one workload per dtype;
#   * nan / inf -- float dtypes over nan / +-inf / huge-magnitude values (the
#     factory copies values verbatim, so the comparison uses equal_nan=True);
#   * negative -- malformed indices / size / values, the wrong layout and a
#     mismatched ``.size_out`` buffer must raise, with the candidate held to
#     the same contract;
#   * broadcast and backward do not apply: this is a pure factory with no
#     arithmetic, no broadcasting semantics and no autograd formula.

# Each 2-D case is (size, indices): ``indices`` is a Python list with one inner
# list per sparse dimension and nnz columns. The column order deliberately
# repeats / reorders coordinates so the constructed tensor is uncoalesced,
# exercising verbatim storage of the raw components.
_COO_2D_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((6, 8), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((5, 6), [[2, 4, 1], [3, 0, 5]]),
    ((3, 3), [[0, 2, 1, 2], [0, 1, 2, 0]]),
]

# Multi-dim cases: (size, indices) where sparse_dim == len(indices) and the
# trailing dims of size are dense (values shape (nnz,) + size[sparse_dim:]).
# Covers 1 sparse dim, dense dims (2 sparse + 1/2 dense) and 3 sparse dims.
_COO_ND_CASES = [
    ((5,), [[0, 2, 4]]),
    ((3, 4, 5), [[0, 1, 2, 1], [1, 3, 0, 2]]),
    ((2, 3, 7), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2], [1, 3, 0]]),
    ((3, 3, 3), [[0, 2, 1], [1, 0, 2], [2, 1, 0]]),
]

# Size-inferred cases: (expected_size, indices, dense_shape). The sparse part
# of the size is max(index[d]) + 1 per sparse dim, the dense part comes from
# the values shape (nnz,) + dense_shape. The last case has a dense dim.
_COO_INFERRED_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]], ()),
    ((5, 6), [[2, 4, 1], [3, 0, 5]], ()),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# Sizes for the size-only overload: the empty sparse tensor (nnz == 0) with
# sparse_dim == len(size) and dense_dim == 0.
_COO_SIZE_ONLY_CASES = [
    ((5,),),
    ((2, 3),),
    ((4, 5, 6),),
]

# Explicit-size empty cases: (size, sparse_dim). nnz == 0 but the logical shape
# and the dense dims are still carried by the tensor.
_COO_EMPTY_CASES = [
    ((2, 3), 2),
    ((4, 5, 6), 2),
    ((2, 3, 4, 5), 2),
    ((3, 4, 5), 3),
]

# Value-range sweep subset: (variant, size, indices, dense_shape) covering the
# explicit-size 2-D path, the explicit-size ND path with dense dims and the
# size-inferred path. For the inferred variant ``size`` is the expected result
# shape (sparse dims are max(index[d]) + 1, dense dims come from the values).
_COO_VALUE_CASES = [
    ("indices_size", (2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ("indices_size", (2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]], (4, 5)),
    ("indices", (2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

# nan / inf / huge-magnitude cases: (size, indices).
_NAN_INF_CASES = [
    ((5,), [[0, 2, 4]]),
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5, 6), [[0, 1, 1], [2, 0, 2]]),
]

_NAN_INF_PATTERN = [
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    -0.0,
    1e30,
    -1e30,
    1.5,
]


# Probe the storage dtypes the factory actually accepts on the active device
# (spec: never guess). ``tu.supported_dtypes`` cannot probe this operator -- its
# default probe calls ``packet.default(x)`` on a dense tensor, which is not how
# ``sparse_coo_tensor`` is invoked -- so a local probe builds a real
# (indices, values) pair instead.
_REQUIRED_CANDIDATE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.int16,
    torch.bool,
]


def _coo_dtype_supported(dtype):
    try:
        indices = torch.zeros((2, 2), dtype=torch.long, device=flag_gems.device)
        values = torch.zeros(2, dtype=dtype, device=flag_gems.device)
        torch.ops.aten.sparse_coo_tensor(
            indices, values, [4, 4], dtype=dtype, device=flag_gems.device
        )
    except Exception:
        return False
    return True


_COO_DTYPES = [
    dtype for dtype in _REQUIRED_CANDIDATE_DTYPES if _coo_dtype_supported(dtype)
]
_FLOAT_COO_DTYPES = [dtype for dtype in _COO_DTYPES if dtype.is_floating_point]
_EXACT_COO_DTYPES = [dtype for dtype in _COO_DTYPES if not dtype.is_floating_point]


def _reference_device():
    # The reference runs on the CPU only under ``--ref cpu``; otherwise it runs
    # on the active device, exactly like the candidate.
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _range_supported(dtype, value_range):
    try:
        tu.make_input(dtype, (4,), value_range)
    except Exception:
        return False
    return True


def _value_range_cases():
    # One pytest case per (range, layout, dtype): uint8 cannot represent the
    # spec's [-1, 0] range, so that pair is filtered out at collection time and
    # every remaining pair is a real workload.
    cases = []
    for case in _COO_VALUE_CASES:
        for dtype in _COO_DTYPES:
            for value_range in tu.selected_ranges():
                if not _range_supported(dtype, value_range):
                    continue
                cases.append((value_range, case, dtype))
    return cases


def _make_index_tensor(indices):
    return torch.tensor(indices, dtype=torch.long, device=flag_gems.device)


def _make_values(nnz, dense_shape, dtype, value_range=None):
    # Stored values come from the shared value-range framework (tu.make_input):
    # range-bound symbols resolve per-dtype, so every storage dtype gets valid
    # inputs within the requested numeric range. Construction copies the raw
    # entries verbatim, so any representable value round-trips exactly.
    if value_range is None:
        value_range = ["-1", "1"]
    shape = (nnz,) + tuple(dense_shape)
    return tu.make_input(dtype, shape, value_range).to(flag_gems.device)


def _make_nan_inf_values(shape, dtype):
    # Repeating pattern of nan / +inf / -inf / huge magnitudes. The factory
    # stores the entries verbatim, so the pattern is fully determined by the
    # value multiset and the reference and the candidate agree under
    # equal_nan=True (fp8_e4m3fn additionally collapses inf / 1e30 to nan, and
    # both sides observe the very same cast tensor).
    base = torch.tensor(_NAN_INF_PATTERN, dtype=torch.float64)
    numel = 1
    for extent in shape:
        numel *= int(extent)
    repeats = (numel + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:numel].reshape(shape).to(dtype).to(flag_gems.device)


def _make_out_buffer(size, dtype, device, nnz):
    # A sparse COO buffer of exactly the requested logical shape. nnz > 0 makes
    # it "dirty" so ``.size_out`` has to reset the stored entries to zero.
    if nnz == 0:
        return torch.ops.aten.sparse_coo_tensor(list(size), dtype=dtype, device=device)
    indices = torch.zeros((len(size), nnz), dtype=torch.long, device=device)
    values = tu.make_input(dtype, (nnz,), ["0", "1"]).to(device)
    return torch.ops.aten.sparse_coo_tensor(
        indices, values, list(size), dtype=dtype, device=device
    )


def _assert_coo_structure(
    res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim, is_coalesced=None
):
    # Structural checks independent of the stored values: layout, shape, dtype,
    # device, sparse/dense split, the nnz count and the coalesced flag.
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert tuple(res_out.shape) == tuple(size)
    assert tuple(ref_out.shape) == tuple(size)
    assert res_out.dtype == dtype
    assert ref_out.dtype == dtype
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == dense_dim
    assert ref_out.sparse_dim() == sparse_dim
    assert ref_out.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(res_out) == nnz
    assert torch.ops.aten._nnz(ref_out) == nnz
    assert tuple(torch.ops.aten._indices(res_out).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._values(res_out).shape) == (nnz,) + tuple(
        size[sparse_dim:]
    )
    # The constructor records the coalesced flag verbatim, so the candidate
    # must match the reference exactly (uncoalesced for nnz > 0, coalesced for
    # nnz == 0 unless is_coalesced=True is passed).
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    if is_coalesced is not None:
        assert res_out.is_coalesced() == is_coalesced
        assert ref_out.is_coalesced() == is_coalesced
    # The index tensors are exact integer data stored verbatim.
    utils.gems_assert_equal(
        torch.ops.aten._indices(res_out), torch.ops.aten._indices(ref_out)
    )


# torch.isclose -- which ``torch.testing.assert_close`` uses internally -- has
# no fp8 CUDA kernel ("mul_cuda" not implemented for 'Float8_e4m3fn'), so fp8
# storages are upcast to float32 before the tolerance comparison. Both sides
# receive the identical input tensor, so the upcast faithfully represents the
# stored bytes.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _comparable(tensor, dtype):
    return tensor.float() if dtype in _FP8_DTYPES else tensor


def _compare_dtype(dtype):
    return torch.float32 if dtype in _FP8_DTYPES else dtype


def _assert_coo_values(res_out, ref_out, dtype, equal_nan=False):
    # The factory performs no arithmetic: float storages compare with the usual
    # tolerance, exact storages must match bit-for-bit.
    if dtype.is_floating_point:
        utils.gems_assert_close(
            _comparable(res_out, dtype),
            _comparable(ref_out, dtype),
            _compare_dtype(dtype),
            equal_nan=equal_nan,
        )
    else:
        utils.gems_assert_equal(res_out, ref_out)


def _resolve_gems_op():
    """Resolve the candidate inside the test (never at import time).

    Resolution order: (1) the process-local override installed by KernelGen,
    (2) the direct ``flag_gems.sparse_coo_tensor`` callable, (3) ``None`` when
    neither exists yet -- the tests then run the reference callable with
    identical call semantics on the active device, so the file stays runnable
    before an implementation is merged.
    """
    try:
        return flag_gems.testing.resolve_gems_op(
            "sparse_coo_tensor", getattr(flag_gems, "sparse_coo_tensor", None)
        )
    except LookupError:
        return None


def _resolve_gems_op_out():
    # The .out overload is a distinct operator with its own public name.
    try:
        return flag_gems.testing.resolve_gems_op(
            "sparse_coo_tensor.size_out",
            getattr(flag_gems, "sparse_coo_tensor_size_out", None),
        )
    except LookupError:
        return None


def _call_reference(indices, values, size, dtype):
    # Mirrors the candidate call exactly. size=None selects the size-inferred
    # ``indices`` overload, a list selects the explicit ``indices_size``
    # overload; the component tensors are cloned and moved to the reference
    # device so a mutating candidate cannot hide behind shared storage.
    ref_indices = utils.to_reference(indices.clone())
    ref_values = utils.to_reference(values.clone())
    if size is None:
        return torch.ops.aten.sparse_coo_tensor(
            ref_indices, ref_values, dtype=dtype, device=ref_indices.device
        )
    return torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )


def _call_candidate(indices, values, size, dtype, **extra):
    op = _resolve_gems_op()
    call = op if op is not None else torch.ops.aten.sparse_coo_tensor
    if size is None:
        return call(indices, values, dtype=dtype, device=indices.device, **extra)
    return call(
        indices, values, list(size), dtype=dtype, device=indices.device, **extra
    )


def _call_candidate_size_only(size, dtype, **extra):
    op = _resolve_gems_op()
    call = op if op is not None else torch.ops.aten.sparse_coo_tensor
    return call(list(size), dtype=dtype, device=flag_gems.device, **extra)


def _call_candidate_size_out(size, out):
    op = _resolve_gems_op_out()
    call = op if op is not None else torch.ops.aten.sparse_coo_tensor.size_out
    return call(list(size), out=out)


def _assert_rejected(ref_call, candidate_call):
    # The reference must raise and the candidate is held to the same contract;
    # NotImplementedError is a RuntimeError subclass, and a candidate that
    # forgets the overload surfaces as TypeError / ValueError.
    with pytest.raises(RuntimeError):
        ref_call()
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        candidate_call()


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_size(case, dtype):
    # Size-only overload: an empty sparse tensor (nnz == 0, coalesced) with the
    # requested logical shape and storage dtype.
    (size,) = case
    ref_device = _reference_device()
    ref_out = torch.ops.aten.sparse_coo_tensor(
        list(size), dtype=dtype, device=ref_device
    )
    res_out = _call_candidate_size_only(size, dtype)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, len(size), 0)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor_size_out
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_size_out(case, dtype):
    # .size_out overload: writes the empty tensor into the caller-provided
    # buffer (resetting a dirty nnz > 0 buffer to nnz == 0) and returns it.
    (size,) = case
    ref_device = _reference_device()
    ref_out = _make_out_buffer(size, dtype, ref_device, nnz=2)
    out = _make_out_buffer(size, dtype, flag_gems.device, nnz=2)

    ref_ret = torch.ops.aten.sparse_coo_tensor.size_out(list(size), out=ref_out)
    res_ret = _call_candidate_size_out(size, out)

    assert ref_ret is ref_out
    assert res_ret is out
    _assert_coo_structure(res_ret, ref_ret, size, 0, dtype, len(size), 0)
    _assert_coo_values(res_ret, ref_ret, dtype)
    # The returned tensor aliases the caller buffer (asserted above), so the
    # buffer itself already carries the reference structure.
    assert tuple(out.shape) == tuple(size)
    assert torch.ops.aten._nnz(out) == 0


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_2D_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size(case, dtype):
    # Explicit-size overload (2 sparse dims, no dense dims). The components are
    # stored verbatim, so duplicate / unsorted coordinates stay uncoalesced.
    size, indices = case
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, (), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_ND_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_nd(case, dtype):
    # Multi-dim overload coverage: 1 sparse dim, dense dims (sparse_dim == 2)
    # and 3 sparse dims.
    size, indices = case
    sparse_dim = len(indices)
    dense_dim = len(size) - sparse_dim
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, tuple(size[sparse_dim:]), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices(case, dtype):
    # Size-inferred overload: the sparse part of the size is max(index[d]) + 1
    # per sparse dim and the dense part comes from the values shape.
    size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype)

    ref_out = _call_reference(indices_t, values, None, dtype)
    res_out = _call_candidate(indices_t, values, None, dtype)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_empty(case, dtype):
    # Explicit-size overload with nnz == 0: the index / value storage is empty
    # but the requested shape, dense dims and dtype are carried by the tensor.
    size, sparse_dim = case
    dense_shape = tuple(size[sparse_dim:])
    indices_t = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, dense_shape, dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(
        res_out, ref_out, size, 0, dtype, sparse_dim, len(dense_shape)
    )
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_is_coalesced(dtype):
    # An explicit is_coalesced=True kwarg is honored by the constructor even
    # when the stored coordinates contain duplicates (the flag is recorded
    # verbatim); the result must be coalesced like the reference.
    size = (2, 3)
    indices = [[0, 1, 1], [2, 0, 2]]
    nnz = 3
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, (), dtype)
    ref_indices = utils.to_reference(indices_t.clone())
    ref_values = utils.to_reference(values.clone())

    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_indices.device,
        is_coalesced=True,
    )
    res_out = _call_candidate(indices_t, values, size, dtype, is_coalesced=True)

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, 2, 0, is_coalesced=True)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("value_range,case,dtype", _value_range_cases())
def test_sparse_coo_tensor_value_ranges(value_range, case, dtype):
    # Value-range sweep over the full dtype contract: construction copies the
    # stored values verbatim, so every range (including the dtype-extreme
    # [0, max] / [min, 0] ranges) must round-trip exactly. Float storages use
    # the tolerance policy, exact storages are compared bit-for-bit.
    variant, size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype, value_range)

    ref_out = _call_reference(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )
    res_out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    _assert_coo_structure(res_out, ref_out, size, nnz, dtype, sparse_dim, dense_dim)
    _assert_coo_values(res_out, ref_out, dtype)
    # The shared helper does its own cpu-side comparison; fp8 is already covered
    # by ``_assert_coo_values`` above and cannot go through torch.isclose.
    if dtype not in _FP8_DTYPES:
        tu.assert_result_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _NAN_INF_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_COO_DTYPES)
def test_sparse_coo_tensor_nan_inf(case, dtype):
    # The factory copies the raw stored values and performs no arithmetic on
    # them, so inf / -inf / nan / -0.0 and huge 1e30 magnitudes survive the
    # construction unchanged (1e30 covers the overflow-to-inf path in
    # fp16/bf16). equal_nan tolerates the nan outputs in every comparison.
    size, indices = case
    sparse_dim = len(indices)
    dense_shape = tuple(size[sparse_dim:])
    nnz = len(indices[0])
    indices_t = _make_index_tensor(indices)
    values = _make_nan_inf_values((nnz,) + dense_shape, dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(
        res_out, ref_out, size, nnz, dtype, sparse_dim, len(dense_shape)
    )
    _assert_coo_values(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(
        _comparable(torch.ops.aten._values(res_out), dtype),
        _comparable(torch.ops.aten._values(ref_out), dtype),
        _compare_dtype(dtype),
        equal_nan=True,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_zero_extent(dtype):
    # A logical size with a zero extent is valid: the result is an nnz == 0
    # coalesced tensor that still carries the full logical shape.
    size = (0, 4)
    indices_t = torch.empty(2, 0, dtype=torch.long, device=flag_gems.device)
    values = _make_values(0, (), dtype)

    ref_out = _call_reference(indices_t, values, size, dtype)
    res_out = _call_candidate(indices_t, values, size, dtype)

    _assert_coo_structure(res_out, ref_out, size, 0, dtype, 2, 0)
    _assert_coo_values(res_out, ref_out, dtype)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_VALUE_CASES)
def test_sparse_coo_tensor_inputs_not_mutated(case):
    # Construction returns a fresh tensor and must not modify the (indices,
    # values) it was handed.
    variant, size, indices, dense_shape = case
    nnz = len(indices[0])
    dtype = torch.float32
    indices_t = _make_index_tensor(indices)
    values = _make_values(nnz, dense_shape, dtype)
    # Snapshot through the reference-device helper: under ``--ref cpu`` the
    # snapshots live on the CPU, which is the convention the accuracy helpers
    # expect for the reference operand.
    indices_before = utils.to_reference(indices_t.clone())
    values_before = utils.to_reference(values.clone())

    out = _call_candidate(
        indices_t, values, size if variant == "indices_size" else None, dtype
    )

    assert out is not indices_t
    utils.gems_assert_equal(indices_t, indices_before)
    utils.gems_assert_close(values, values_before, dtype)


# ---------------------------------------------------------------------------
# Negative cases: malformed inputs must raise, and the candidate is held to the
# same contract as the reference.
# ---------------------------------------------------------------------------


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_ndim():
    # indices must be 2-D (sparse_dim, nnz); a 1-D index tensor is rejected.
    indices_t = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(3, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [3], torch.float32),
        lambda: _call_candidate(indices_t, values, [3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size():
    # A negative logical size is rejected (numel overflow); the candidate must
    # fail too rather than accept a nonsensical shape.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [-2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [-2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size_only_size():
    # The same negative-size rule applies to the size-only overload.
    ref_device = _reference_device()

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [-2, 3], dtype=torch.float32, device=ref_device
        ),
        lambda: _call_candidate_size_only([-2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_non_integer_size():
    # The size must be an int[]; a float size matches no schema and is rejected.
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [2.5, 3], dtype=torch.float32, device=_reference_device()
        ),
        lambda: _call_candidate_size_only([2.5, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_dtype():
    # The sparse COO layout requires int64 indices; an int32 index tensor is
    # rejected by the reference and must be by the candidate too.
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.int32, device=flag_gems.device
    )
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_float():
    # A floating-point index tensor is likewise rejected.
    indices_t = _make_index_tensor([[0, 1], [2, 0]]).float()
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_nnz_mismatch():
    # indices and values must carry the same number of entries; a mismatch is
    # rejected by the reference and must be by the candidate too.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(3, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_values_dense_dims():
    # The values tensor must have exactly sparse_dim + dense_dim dims; a values
    # tensor with the wrong rank is rejected.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2, 3, 4], torch.float32),
        lambda: _call_candidate(indices_t, values, [2, 3, 4], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_size_rank():
    # ``indices`` carries one row per sparse dim; a size with a different rank
    # cannot be reconciled and is rejected.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, [2], torch.float32),
        lambda: _call_candidate(indices_t, values, [2], torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_inferred_negative_index():
    # The size-inferred overload derives each sparse dim from max(index) + 1,
    # so a negative coordinate cannot be resolved and is rejected.
    indices_t = torch.tensor([[-1, 1]], dtype=torch.long, device=flag_gems.device)
    values = _make_values(2, (), torch.float32)

    _assert_rejected(
        lambda: _call_reference(indices_t, values, None, torch.float32),
        lambda: _call_candidate(indices_t, values, None, torch.float32),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_layout():
    # Only the sparse COO layout is produced by this name; a CSR request is
    # rejected even though the components are otherwise valid.
    indices_t = _make_index_tensor([[0, 1], [2, 0]])
    values = _make_values(2, (), torch.float32)
    ref_indices = utils.to_reference(indices_t.clone())
    ref_values = utils.to_reference(values.clone())

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            ref_indices,
            ref_values,
            [2, 3],
            dtype=torch.float32,
            device=ref_indices.device,
            layout=torch.sparse_csr,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3], torch.float32, layout=torch.sparse_csr
        ),
    )


@pytest.mark.sparse_coo_tensor_size_out
def test_sparse_coo_tensor_size_out_negative_shape():
    # The out buffer must already carry the requested logical shape: a
    # mismatched buffer needs a sparse resize, which has no kernel here, so the
    # overload raises and the candidate must reject it as well.
    ref_out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=_reference_device()
    )
    out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=flag_gems.device
    )

    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor.size_out([2, 3], out=ref_out),
        lambda: _call_candidate_size_out([2, 3], out),
    )
