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

import os
import sys

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# KernelGen's in-process verification (override_gems_op + pytest.main) stages the
# test files into an isolated temp copy of the checkout, where the relative
# ``from . import accuracy_utils`` cannot resolve this checkout's tests package
# through normal package discovery. Put the checkout root on sys.path so the
# ``tests`` package resolves to THIS checkout no matter how pytest is invoked.
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHECKOUT_ROOT not in sys.path:
    sys.path.insert(0, _CHECKOUT_ROOT)

from . import accuracy_utils as utils  # noqa: E402
from . import conftest as cfg  # noqa: E402
from . import test_utils as tu  # noqa: E402

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


# aten::_efficientzerotensor is a factory: given a size (plus optional
# dtype/device) it returns a fresh all-zero tensor. There is no input tensor, so
# the regular-operator spec dimensions adapt as follows:
# - Value ranges: N/A -- there are no input values to vary; the output is
#   bit-exact zero for every supported dtype, so the value dimension here is the
#   dtype family (bool / int / float), each compared with exact equality.
# - Shape levels: tu.selected_shapes() (quick/core/all via TEST_LEVEL).
# - Broadcast: N/A -- the only "input" is a size list; nothing to broadcast.
# - Backward: N/A -- a factory with no autograd support (no differentiable
#   input, output is never a function of another tensor).
# - nan/inf: trivially satisfied -- the output is deterministic zeros that never
#   contain nan/inf and there is no input through which non-finite values could
#   leak.
# - Negative cases: a negative dimension and a non-strided layout must raise on
#   both the aten reference and the candidate (covered below).
_EFFICIENTZEROTENSOR_DTYPES = (
    utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)


@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_zero_fill(shape, dtype):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.ops.aten._efficientzerotensor(shape, dtype=dtype, device=ref_device)

    gems_op = _resolve("_efficientzerotensor")
    res_out = gems_op(shape, dtype=dtype, device=flag_gems.device)

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    # flag_gems.device may carry no index (e.g. 'cuda') while a created
    # tensor reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    # The factory returns a fresh, non-view, all-zero tensor.
    assert not res_out._is_view()
    # The values are exactly zero for every dtype, so exact equality is valid.
    utils.gems_assert_equal(res_out, ref_out)


# aten::_efficientzerotensor.out writes zeros into the provided ``out`` buffer
# and returns the same object (alias semantics).
@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_out(shape, dtype):
    ref_device = "cpu" if cfg.TO_CPU else flag_gems.device
    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out_buf = torch.full(shape, 7, dtype=dtype, device=ref_device)
    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_out_buf)
    assert ref_out is ref_out_buf

    act_out_buf = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)
    gems_op = _resolve("_efficientzerotensor_out")
    res_out = gems_op(shape, out=act_out_buf)
    assert res_out is act_out_buf

    utils.gems_assert_equal(act_out_buf, ref_out_buf)


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_negative_size():
    # A negative dimension is invalid; aten raises RuntimeError and the
    # candidate must reject it too rather than silently truncating.
    with pytest.raises(RuntimeError):
        torch.ops.aten._efficientzerotensor((-1,), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve("_efficientzerotensor")((-1,), dtype=torch.float32)


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_non_strided_layout():
    # Only the strided layout is supported; aten has no sparse kernel and the
    # candidate must reject the request as well.
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._efficientzerotensor(
            (2, 3), dtype=torch.float32, layout=torch.sparse_coo
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_efficientzerotensor")(
            (2, 3), dtype=torch.float32, layout=torch.sparse_coo
        )
