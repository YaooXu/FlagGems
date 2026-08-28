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

# ``_has_same_storage_numel`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._has_same_storage_numel`` and
# ``-m _has_same_storage_numel`` both work.
setattr(
    pytest.mark,
    "_has_same_storage_numel",
    MarkDecorator(
        Mark("_has_same_storage_numel", (), {}, _ispytest=True), _ispytest=True
    ),
)

# aten::_has_same_storage_numel(Tensor self, Tensor other) -> bool compares the
# *storage* element counts of the two tensors (self.storage().numel() ==
# other.storage().numel()), not their logical numel. Views keep the full
# storage of their base, so a (4, 4) row slice still has a 16-element storage
# while an expanded (4, 4) tensor built from a (4, 1) base only has 4. Each
# pair below is a distinct parametrized workload; both True and False outcomes
# are covered, including cases where the logical shapes agree but the storage
# sizes differ.
_HAS_SAME_STORAGE_NUMEL_CASES = [
    pytest.param(("plain", (4, 4)), ("plain", (4, 4)), id="same_shape"),
    pytest.param(("plain", (4, 4)), ("plain", (16,)), id="same_numel_reshaped"),
    pytest.param(("plain", (4, 4)), ("plain", (8,)), id="different_numel"),
    pytest.param(("plain", (4, 4)), ("transposed", (4, 4)), id="transposed_view"),
    pytest.param(("plain", (4, 4)), ("row_view", (4, 4)), id="row_slice_view"),
    pytest.param(("plain", (4, 4)), ("narrowed", (16,)), id="narrowed_same_storage"),
    pytest.param(
        ("plain", (4, 4)), ("expanded", (4, 4)), id="expanded_storage_smaller"
    ),
    pytest.param(("expanded", (4, 4)), ("plain", (4,)), id="expanded_storage_equal"),
    pytest.param(("row_view", (4, 4)), ("plain", (4,)), id="row_slice_storage_larger"),
    pytest.param(("narrowed", (16,)), ("plain", (4,)), id="narrowed_storage_larger"),
    pytest.param(("plain", ()), ("plain", (1,)), id="scalar_vs_single"),
]

# The op is a pure storage-metadata query: it never reads the tensor values, so
# every storage dtype family the runtime supports is exercised.
_HAS_SAME_STORAGE_NUMEL_DTYPES = (
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)


def _make_tensor(spec, dtype, device):
    """Build a tensor with the requested storage-layout spec on ``device``."""
    kind, shape = spec
    if kind == "plain":
        return torch.zeros(shape, dtype=dtype, device=device)
    if kind == "transposed":
        return torch.zeros((shape[1], shape[0]), dtype=dtype, device=device).t()
    if kind == "row_view":
        return torch.zeros(shape, dtype=dtype, device=device)[0]
    if kind == "expanded":
        base = torch.zeros((shape[0], 1), dtype=dtype, device=device)
        return base.expand(shape)
    if kind == "narrowed":
        return torch.zeros(shape, dtype=dtype, device=device).narrow(
            0, shape[0] // 4, shape[0] // 2
        )
    raise ValueError(f"Unknown tensor spec kind: {kind!r}")


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems._has_same_storage_numel is registered; resolution order
    # is: (1) override, (2) the direct flag_gems._has_same_storage_numel
    # callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "_has_same_storage_numel",
        getattr(flag_gems, "_has_same_storage_numel", None),
    )


@pytest.mark._has_same_storage_numel
@pytest.mark.parametrize("self_spec,other_spec", _HAS_SAME_STORAGE_NUMEL_CASES)
@pytest.mark.parametrize("dtype", _HAS_SAME_STORAGE_NUMEL_DTYPES)
def test__has_same_storage_numel(self_spec, other_spec, dtype):
    self_t = _make_tensor(self_spec, dtype, flag_gems.device)
    other_t = _make_tensor(other_spec, dtype, flag_gems.device)

    # Build the reference from the same storage-layout spec on the reference
    # device: moving a view to CPU would compact its storage and change the
    # answer, so both sides must be constructed with identical layouts (see the
    # non-contiguous handling in test__add_batch_dim).
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_self = _make_tensor(self_spec, dtype, ref_device)
    ref_other = _make_tensor(other_spec, dtype, ref_device)

    ref_out = torch.ops.aten._has_same_storage_numel(ref_self, ref_other)
    res_out = _resolve_gems_op()(self_t, other_t)

    assert isinstance(ref_out, bool)
    assert isinstance(res_out, (bool, torch.Tensor))
    utils.gems_assert_equal(
        torch.tensor(res_out, device="cpu"), torch.tensor(ref_out, device="cpu")
    )
