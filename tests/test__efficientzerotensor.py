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
from . import conftest as cfg

# ``_efficientzerotensor`` starts with an underscore, and ``pytest.mark`` refuses
# to generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._efficientzerotensor`` and
# ``-m _efficientzerotensor`` both work.
for _name in ("_efficientzerotensor", "_efficientzerotensor_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )


def _resolve(name):
    # Resolved inside each test (never at import time) so that a process-local
    # override installed by KernelGen via ``override_gems_op`` for this run
    # wins. The default stays None until flag_gems._efficientzerotensor is
    # registered; resolution order is: (1) override, (2) the direct
    # flag_gems._efficientzerotensor callable, (3) LookupError.
    return flag_gems.testing.resolve_gems_op(name, getattr(flag_gems, name, None))


# aten::_efficientzerotensor is a factory that returns an all-zero tensor backed
# by a single shared zero byte (CUDA nbytes == 0). The values are exactly zero,
# so an exact equality comparison is valid for every dtype.
@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test__efficientzerotensor(shape, dtype):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten._efficientzerotensor(shape, dtype=dtype, device=ref_device)

    gems_op = _resolve("_efficientzerotensor")
    res_out = gems_op(shape, dtype=dtype, device=flag_gems.device)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # flag_gems.device may carry no index (e.g. 'cuda') while a created
    # tensor reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    utils.gems_assert_equal(res_out, ref_out)


# aten::_efficientzerotensor.out writes zeros into the provided ``out`` buffer
# and returns the same object (alias semantics).
@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test__efficientzerotensor_out(shape, dtype):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out_buf = torch.ones(shape, dtype=dtype, device=ref_device)
    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_out_buf)
    assert ref_out is ref_out_buf

    act_out_buf = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    gems_op = _resolve("_efficientzerotensor_out")
    res_out = gems_op(shape, out=act_out_buf)
    assert res_out is act_out_buf

    utils.gems_assert_equal(act_out_buf, ref_out_buf)
