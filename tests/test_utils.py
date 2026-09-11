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
    callable ``(op_name) -> bool`` (used by callers that want a custom check).
    The default probe builds a small input for a few ranks/dtypes and calls the
    op, treating any exception as "unsupported".
    """
    import torch

    candidates = list(candidates if candidates is not None else REQUIRED_DTYPES)
    if probe is not None:
        return [d for d in candidates if probe(operator, d)]

    packet = getattr(torch.ops.aten, operator, None)
    if packet is None:
        return []
    supported = []
    for dtype in candidates:
        try:
            x = torch.testing.make_tensor(
                (4,), dtype=dtype, device=flag_gems.device, low=0, high=1
            )
            packet.default(x)
        except Exception:
            continue
        supported.append(dtype)
    return supported
