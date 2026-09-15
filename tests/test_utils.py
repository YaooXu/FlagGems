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


"""Shared test utilities for the regular-operator test spec.

Implements the value-range / shape-level / broadcast / backward conventions
from the "常规算子测试用例" spec (quick / full levels selected by the pytest
``--quick`` flag, matching FlagGems' own accuracy_utils convention). Tests
reference these helpers so the value-range and shape-selection logic lives in
one place instead of being copied into every ``tests/test_<op>.py`` file.

Reference example: the `add` sample (123.py) attached to the spec.
"""

import torch

import flag_gems

from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------
#
# Two levels only, matching FlagGems' ``--quick`` pytest flag:
#   * quick -- smoke subset (``pytest --quick`` sets conftest.QUICK_MODE)
#   * full  -- everything else (default)
# ``LEVEL`` is kept as a derived string so files that branch on it
# (``tu.LEVEL == "quick"`` / ``tu.LEVEL == "all"``) keep working.

LEVEL = "quick" if QUICK_MODE else "all"


def selected_shapes():
    return QUICK_SHAPES if QUICK_MODE else ALL_SHAPES


def selected_ranges():
    return QUICK_RANGES if QUICK_MODE else ALL_RANGES


# ---------------------------------------------------------------------------
# dtype bounds and value-range resolution
# ---------------------------------------------------------------------------


def dtype_bounds(dtype):
    """Return the (min, max) value bounds of ``dtype``.

    - bool: fixed 0/1
    - complex: bounds of its real float dtype
    - floating / integer: finfo / iinfo min/max
    """
    if dtype == torch.bool:
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol (-1 / 0 / 1 / max / min / max/2 / min/2)
    to an actual value for ``dtype``."""
    low, high = dtype_bounds(dtype)
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    ``value_range`` is a [low_symbol, high_symbol] pair whose symbols resolve
    per-dtype (max/min are the dtype bounds). bool ignores the range; integer
    ranges are snapped to ints; a degenerate range (low == high) fills the
    constant; everything else uses torch.testing.make_tensor (complex fills
    both real and imaginary parts).

    Bounds are clamped to the dtype's representable range first, so the spec's
    five ranges work unchanged for dtypes that cannot represent a bound (e.g.
    uint8 cannot hold ``-1``, so ``[-1,0]`` becomes the degenerate ``[0,0]``
    → a constant zero fill). This keeps the caller from having to special-case
    unsigned dtypes.
    """
    low = resolve_bound(value_range[0], dtype)
    high = resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        dtype_min, dtype_max = dtype_bounds(dtype)
        # Clamp into the representable range (e.g. uint8 [-1,0] -> [0,0]).
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def assert_result_close(result, reference):
    """Arithmetic comparison using the tested dtype's shared tolerances."""
    from . import accuracy_utils as utils

    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    if result.is_floating_point() or result.is_complex():
        utils.gems_assert_close(result, reference, result.dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(result, reference)


def assert_result_equal(result, reference):
    from . import accuracy_utils as utils

    utils.gems_assert_equal(result, reference, equal_nan=True)


def _assert_input_snapshot(actual, expected):
    if not isinstance(actual, torch.Tensor):
        if isinstance(actual, (tuple, list)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                _assert_input_snapshot(a, b)
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                _assert_input_snapshot(actual[key], expected[key])
        return
    assert actual.dtype == expected.dtype
    assert actual.layout == expected.layout
    assert actual.device == expected.device
    if actual.is_nested:
        _assert_input_snapshot(actual.unbind(), expected.unbind())
    elif actual.layout == torch.sparse_coo:
        assert actual.shape == expected.shape
        assert actual.is_coalesced() == expected.is_coalesced()
        assert_result_equal(actual._indices(), expected._indices())
        assert_result_equal(actual._values(), expected._values())
    elif actual.layout != torch.strided:
        assert actual.shape == expected.shape
        names = (
            ("crow_indices", "col_indices")
            if actual.layout in (torch.sparse_csr, torch.sparse_bsr)
            else ("ccol_indices", "row_indices")
        )
        for name in names:
            assert_result_equal(getattr(actual, name)(), getattr(expected, name)())
        assert_result_equal(actual.values(), expected.values())
    elif actual.is_quantized:
        assert actual.qscheme() == expected.qscheme()
        assert_result_equal(actual.int_repr(), expected.int_repr())
        if actual.qscheme() == torch.per_tensor_affine:
            assert actual.q_scale() == expected.q_scale()
            assert actual.q_zero_point() == expected.q_zero_point()
        else:
            assert_result_equal(
                actual.q_per_channel_scales(), expected.q_per_channel_scales()
            )
            assert_result_equal(
                actual.q_per_channel_zero_points(), expected.q_per_channel_zero_points()
            )
            assert actual.q_per_channel_axis() == expected.q_per_channel_axis()
    else:
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.is_conj() == expected.is_conj()
        assert actual.is_neg() == expected.is_neg()
        # Comparing the unchanged physical values avoids materializing a lazy
        # negative FP8/bool view, whose eager neg kernel may not exist.
        if actual.is_neg():
            actual, expected = torch._neg_view(actual), torch._neg_view(expected)
        assert_result_equal(actual, expected)


def resolve_gems_op(operator, default=None, *, expected_input_metadata=None):
    """Resolve one public candidate and verify its non-mutating inputs.

    In-place operators may change self; out forms may change their explicit
    output buffers. Snapshots are independent from both the candidate and oracle.
    """
    candidate = flag_gems.testing.resolve_gems_op(operator, default)

    def checked(*args, **kwargs):
        inputs = args[1:] if operator.endswith("_") else args
        outputs = {
            "out",
            "output",
            "grad_input",
            "grad_weight",
            "grad_bias",
            "out0",
            "out1",
            "out2",
        }
        keywords = {k: v for k, v in kwargs.items() if k not in outputs}

        # Explicit output aliases are allowed to change; other inputs are not.
        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from tensors(item)

        mutable = list(tensors(args[:1])) if operator.endswith("_") else []
        for key in outputs:
            mutable.extend(tensors(kwargs.get(key)))

        def protected(value):
            if isinstance(value, torch.Tensor):
                if any(torch._C._is_alias_of(value, out) for out in mutable):
                    return None
            elif isinstance(value, tuple):
                return tuple(protected(x) for x in value)
            elif isinstance(value, list):
                return [protected(x) for x in value]
            elif isinstance(value, dict):
                return {k: protected(v) for k, v in value.items()}
            return value

        inputs, keywords = protected((inputs, keywords))
        snapshot = flag_gems.testing.clone_inputs((inputs, keywords))
        if expected_input_metadata:
            # Some native contracts alter input metadata. Callers must obtain
            # this expectation from the independently executed reference;
            # values remain protected by the original snapshot.
            expected_args = list(snapshot[0])
            for index, reference in expected_input_metadata.items():
                old = expected_args[index]
                assert old.numel() == reference.numel()
                expected_args[index] = old.as_strided(
                    reference.shape, reference.stride(), old.storage_offset()
                )
            snapshot = (tuple(expected_args), snapshot[1])
        result = candidate(*args, **kwargs)
        _assert_input_snapshot((inputs, keywords), snapshot)
        return result

    return checked


# ---------------------------------------------------------------------------
# Shapes and value ranges by level
# ---------------------------------------------------------------------------

QUICK_SHAPES = [
    (2, 19, 7),
]

# Full set — the required 7 shapes from the team's operator-test spec
# (0~5 dims; a fixed-dim operator keeps only the ranks it accepts).
REQUIRED_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]
ALL_SHAPES = REQUIRED_SHAPES

QUICK_RANGES = [
    ["-1", "1"],
]

# Full set — the required 5 value ranges from the spec
#   [-1,1], [0,1], [-1,0], [0,dtype_max], [dtype_min,0]
REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]
ALL_RANGES = REQUIRED_RANGES

# Required dtype coverage (spec). int8/uint8/fp8 must be present for every
# operator whose CUDA kernel supports them; fp32/bf16/fp16/int32/int64 are
# added when the operator supports them.
REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

# Required minimum number of correctness cases per operator (spec: "at least
# 100"). 5 ranges x 7 shapes = 35 per dtype, so >=3 dtypes already clears it.
MIN_CASES = 100


def supported_dtypes(operator, candidates=None, probe=None):
    """Return the subset of ``candidates`` that the op supports on the device.

    ``operator`` is a ``torch.ops.aten`` operator name; ``probe`` is an optional
    callable ``(op_name, dtype) -> bool`` (used by callers that want a custom check).
    The default probe builds a small rank-1 input for each dtype and calls the
    op. Unexpected errors are inconclusive and must not remove a dtype.
    """
    import torch

    candidates = list(candidates if candidates is not None else REQUIRED_DTYPES)
    if probe is not None:
        return [d for d in candidates if probe(operator, d)]

    packet = getattr(torch.ops.aten, operator, None)
    if packet is None:
        raise ValueError(f"Unknown operator: {operator}")
    supported = []
    for dtype in candidates:
        # Input-construction errors never count as operator capability evidence.
        x = torch.testing.make_tensor(
            (4,), dtype=dtype, device=flag_gems.device, low=0, high=1
        )
        try:
            packet.default(x)
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if any(
                text in message
                for text in (
                    "not implemented for",
                    "not supported for",
                    "unsupported dtype",
                )
            ):
                continue
            raise RuntimeError(
                f"Inconclusive dtype probe for {operator}/{dtype}; supply a valid probe"
            ) from exc
        supported.append(dtype)
    return supported


def special_value_cases(dtypes):
    """Representable special scenarios, kept separate in collected case IDs."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        if dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


def make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)


def is_extreme_range(value_range):
    return any(bound in {"min", "max", "min/2", "max/2"} for bound in value_range)
