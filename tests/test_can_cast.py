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

# aten::can_cast(ScalarType from_, ScalarType to) -> bool answers whether a value
# of ScalarType ``from_`` can be cast to ScalarType ``to`` according to aten's
# cast-safety rules. It is a pure dtype-metadata query: no tensor is created,
# the device is never touched, and the result is a plain Python bool. The full
# cross product over the standard scalar types covers every family pair
# (bool/integral/floating/complex) in both directions, including the diagonal
# (every type can cast to itself) and the asymmetric cases such as float -> int.
_CAN_CAST_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]
if hasattr(torch, "complex32"):
    _CAN_CAST_DTYPES.append(torch.complex32)
_CAN_CAST_DTYPES += [torch.complex64, torch.complex128]


def _resolve_gems_op():
    # Resolved inside each test (never at import time) so that the process-local
    # override installed by KernelGen for this run wins. The default stays None
    # until flag_gems.can_cast is registered; resolution order is: (1) override,
    # (2) the direct flag_gems.can_cast callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(
        "can_cast", getattr(flag_gems, "can_cast", None)
    )


@pytest.mark.can_cast
@pytest.mark.parametrize("from_dtype", _CAN_CAST_DTYPES)
@pytest.mark.parametrize("to_dtype", _CAN_CAST_DTYPES)
def test_can_cast(from_dtype, to_dtype):
    ref_out = torch.ops.aten.can_cast(from_dtype, to_dtype)
    res_out = _resolve_gems_op()(from_dtype, to_dtype)

    assert isinstance(ref_out, bool)
    assert isinstance(res_out, (bool, torch.Tensor))
    utils.gems_assert_equal(
        torch.tensor(res_out, device="cpu"), torch.tensor(ref_out, device="cpu")
    )
