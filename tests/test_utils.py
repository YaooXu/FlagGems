"""Shared test utilities for the regular-operator test spec.

Implements the value-range / shape-level / broadcast / backward conventions
from the "常规算子测试用例" spec (quick / core / all levels selected by the
``TEST_LEVEL`` environment variable). Tests reference these helpers so the
value-range and shape-selection logic lives in one place instead of being
copied into every ``tests/test_<op>.py`` file.

Reference example: the `add` sample (123.py) attached to the spec.
"""

import os

import torch

# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------

LEVEL = os.getenv("TEST_LEVEL", "core")


def selected_shapes():
    if LEVEL == "quick":
        return QUICK_SHAPES
    if LEVEL in ("all", "extended"):
        return ALL_SHAPES
    return CORE_SHAPES


def selected_ranges():
    if LEVEL == "quick":
        return QUICK_RANGES
    if LEVEL in ("all", "extended"):
        return ALL_RANGES
    return CORE_RANGES


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
        real = torch.float32 if dtype == torch.complex64 else torch.float64
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
    """
    low = resolve_bound(value_range[0], dtype)
    high = resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=DEVICE).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)

    if low == high:
        return torch.full(shape, low, device=DEVICE, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=DEVICE, low=low, high=high
    )


def assert_result_close(result, reference):
    """Compare result vs reference with value-range-friendly tolerances.

    Integer/bool must match bit-exactly; float/complex use rtol=1e-2 atol=1e-3
    with equal_nan=True (so inf + (-inf) = nan cases pass).
    """
    result_cpu = result.detach().cpu()
    reference_cpu = reference.detach().cpu()

    if result_cpu.dtype == torch.bool or not (
        result_cpu.is_floating_point() or result_cpu.is_complex()
    ):
        torch.testing.assert_close(result_cpu, reference_cpu, rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            result_cpu,
            reference_cpu,
            rtol=1e-2,
            atol=1e-3,
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# Shapes and value ranges by level
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUICK_SHAPES = [
    (2, 19, 7),
]

CORE_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (7, 13, 29),  # regular 3-dim
]

ALL_SHAPES = CORE_SHAPES + [
    (16, 7, 57, 32, 29),  # 5-dim
    (12, 9, 3, 6, 8, 6),  # 6-dim
    (3, 6, 4, 4, 6, 5, 4),  # 7-dim
    (7, 3, 12, 4, 2, 15, 2, 2),  # 8-dim
]

QUICK_RANGES = [
    ["-1", "1"],
]

CORE_RANGES = QUICK_RANGES + [
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]

ALL_RANGES = CORE_RANGES + [
    ["0", "max/2"],
    ["min/2", "0"],
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
]
