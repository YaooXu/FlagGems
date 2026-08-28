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

import sys
from pathlib import Path

import pytest
import torch

import flag_gems


def _unshadow_packages():
    """Realign sys.path / sys.modules so the ``tests`` and ``benchmark`` packages
    that physically contain this file win over same-named packages on the process
    sys.path.

    The KernelGen TestWriter harness stages a temporary copy of the FlagGems tree
    and runs ``pytest --import-mode=importlib`` from the kernelgen repo process,
    whose sys.path contains the kernelgen root (which ships its own ``tests/``
    package). That package would otherwise shadow the real one and break the
    relative imports below. Leaving the real package root at ``sys.path[0]`` also
    lets the benchmark phase of the same process import its ``benchmark`` package
    the same way.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    for name in ("tests", "benchmark"):
        real = pkg_root / name
        if not real.is_dir():
            continue
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None):
            try:
                if Path(mod.__file__).resolve().parent == real.resolve():
                    continue  # already the real package
            except Exception:
                pass
        prefix = name + "."
        for key in [
            k
            for k in sys.modules
            if (k == name or k.startswith(prefix)) and k != __name__
        ]:
            sys.modules.pop(key, None)
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))


_unshadow_packages()

from . import accuracy_utils as utils  # noqa: E402

# aten::coalesce(Tensor(a) self) -> Tensor(a) is the public method variant of
# sparse-tensor coalescing. It merges duplicate entries of an uncoalesced
# sparse COO tensor: the result has unique, lexicographically sorted indices
# and values equal to the sum of the entries sharing each index. Unlike
# aten::_coalesce (which requires an uncoalesced input), coalesce also accepts
# an already-coalesced tensor and, per its ``Tensor(a)`` alias annotation,
# returns the input itself unchanged. Every uncoalesced case below picks
# nnz > numel so the pigeonhole principle guarantees duplicate indices and
# coalescing always has work to do; coalesce never mutates its input.
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
]

# The identity branch (input already coalesced -> returns self) is exercised on
# a subset of the shapes above; the input is made coalesced via .coalesce().
_COALESCED_CASES = [
    ((5, 5), 30),
    ((3, 5, 7), 120),
]

# Summing duplicates is exact for every integer/bool storage dtype aten
# supports; float summation order may differ between implementations, so float
# results are compared with the dtype-appropriate tolerance and integer/bool
# results with exact equality.
_COALESCE_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES


def _make_input(shape, nnz, dtype):
    # Deterministic CPU-side generation (then the sparse tensor is created on
    # the test device). Index rows are drawn with replacement, so duplicates
    # are guaranteed whenever nnz > numel.
    gen = torch.Generator("cpu").manual_seed(2026)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    if dtype.is_floating_point:
        # Non-negative values: coalescing sums duplicates, and summing in
        # different orders can differ by ulps in fp16/bf16. With same-sign
        # values there is no cancellation, so any summation order stays within
        # the dtype tolerance (this keeps the CPU reference and the device
        # candidate comparable in --ref cpu mode).
        values = torch.rand(nnz, dtype=dtype, generator=gen)
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    else:
        # Keep the magnitude small so summed duplicates cannot overflow the
        # smallest integer dtype (int16).
        values = torch.randint(-5, 6, (nnz,), dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


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
    # Indices are int64 and must match exactly (unique, sorted index tuples).
    utils.gems_assert_equal(res_out.indices(), ref_out.indices())
    # Values are summed duplicates; use the dtype-appropriate tolerance.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    else:
        utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test_coalesce(case, dtype):
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.coalesce(ref_inp)
    res_out = _resolve_gems_op()(inp)

    _assert_coalesced(res_out, ref_out, dtype)
    # coalesce returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()
    assert not ref_inp.is_coalesced()


@pytest.mark.coalesce_coalesced
@pytest.mark.parametrize("case", _COALESCED_CASES)
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
