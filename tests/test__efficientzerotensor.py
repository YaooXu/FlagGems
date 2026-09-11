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
from . import test_utils as tu

# ``_efficientzerotensor`` starts with an underscore, and ``pytest.mark``
# refuses to create a marker through attribute access for such names. Register
# the markers on the MarkGenerator directly so both
# ``@pytest.mark._efficientzerotensor`` and ``-m _efficientzerotensor`` work.
for _name in ("_efficientzerotensor", "_efficientzerotensor_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# ``aten::_efficientzerotensor`` is a *factory*: given a size (plus optional
# dtype/device) it returns a fresh all-zero tensor. The regular-operator spec
# dimensions adapt as follows:
# - Value ranges -- there is no input tensor to vary, so the range framework is
#   applied to the only data the operator touches: the pre-existing contents of
#   the ``out`` buffer of the ``.out`` overload, which must be overwritten with
#   zeros. ``tu.make_input`` fills that buffer with each of the five spec
#   ranges, and one further test fills it with a fixed non-zero sentinel so a
#   missing write is always detected.
# - Shape levels -- ``tu.selected_shapes()`` (quick/all via ``--quick``) plus a
#   zero-sized-shape boundary set.
# - Dtypes -- bool / int / float, including float8 / int8 / uint8 when the
#   active backend accepts them (probed, never guessed); every value comparison
#   is exact because the output is bit-exact zero.
# - Broadcast -- N/A, the only "input" is a size list; nothing to broadcast.
# - Backward -- N/A, a factory has no differentiable input and its output is not
#   a function of another tensor.
# - nan/inf -- trivially satisfied, the output is deterministic zeros that can
#   never contain nan/inf and there is no input through which non-finite values
#   could leak.
# - Negative cases -- a negative dimension, a non-strided layout and a
#   non-integer size element must all be rejected.


def _resolve(name):
    """Resolve a KernelGen override, or the direct FlagGems callable.

    Resolution happens inside each test (never at import time) so a
    process-local override installed via ``override_gems_op`` wins.
    """
    aliases = [name]
    if name.endswith("_out"):
        # KernelGen may register the ``.out`` overload under either the
        # FlagGems ``<op>_out`` convention or the dotted aten name.
        aliases.append(name[: -len("_out")] + ".out")
    last_error = None
    for alias in aliases:
        try:
            return flag_gems.testing.resolve_gems_op(
                alias, getattr(flag_gems, alias, None)
            )
        except LookupError as exc:
            last_error = exc
    raise last_error


def _reference_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _unique(dtypes):
    seen = set()
    ordered = []
    for dtype in dtypes:
        if dtype not in seen:
            seen.add(dtype)
            ordered.append(dtype)
    return ordered


def _probe_dtype(op_name, dtype):
    # The factory takes a size list rather than an input tensor, which the
    # shared ``tu.supported_dtypes`` default probe cannot model, so a custom
    # probe is supplied.
    try:
        getattr(torch.ops.aten, op_name)((2, 3), dtype=dtype, device=flag_gems.device)
        return True
    except Exception:
        return False


_DTYPE_CANDIDATES = _unique(
    tu.REQUIRED_DTYPES
    + utils.BOOL_TYPES
    + utils.ALL_INT_DTYPES
    + utils.ALL_FLOAT_DTYPES
)

# On CUDA every required dtype (int8 / uint8 / float8_e4m3fn / float8_e5m2 /
# float32 / bfloat16 / float16 / int32 / int64) is accepted; backends that
# cannot represent some of them simply drop those parametrizations. If the probe
# yields nothing, keep the full candidate list rather than a float32-only
# fallback, so a failed/absent probe never silently drops the spec-required
# int8/uint8/fp8 dtypes.
_EFFICIENTZEROTENSOR_DTYPES = tu.supported_dtypes(
    "_efficientzerotensor",
    candidates=_DTYPE_CANDIDATES,
    probe=_probe_dtype,
) or list(_DTYPE_CANDIDATES)


# One (dtype, value_range) pair per Workload: this crosses the five spec ranges
# with every supported dtype without hiding cases inside a loop. The negative
# ranges are realised for every dtype because ``tu.make_input`` clamps a bound
# the dtype cannot represent (e.g. ``[-1, 0]`` on uint8) into the dtype's range
# and fills that constant.
_OUT_RANGE_PARAMS = [
    (dtype, value_range)
    for dtype in _EFFICIENTZEROTENSOR_DTYPES
    for value_range in tu.selected_ranges()
]

# Zero-element boundary shapes (rank 1 to 3); the factory must still report the
# requested shape and return exactly zero elements.
_ZERO_SIZE_SHAPES = [(0,), (0, 3), (2, 0, 4), (0, 0)]


@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_zero_fill(shape, dtype):
    ref_out = torch.ops.aten._efficientzerotensor(
        shape, dtype=dtype, device=_reference_device()
    )

    gems_op = _resolve("_efficientzerotensor")
    res_out = gems_op(shape, dtype=dtype, device=flag_gems.device)

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    # flag_gems.device may carry no index (e.g. 'cuda') while a fresh tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    # The factory returns a fresh, non-view, all-zero tensor.
    assert not res_out._is_view()
    # Values are exactly zero for every dtype, so exact equality is valid.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", _ZERO_SIZE_SHAPES)
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_zero_size(shape, dtype):
    ref_out = torch.ops.aten._efficientzerotensor(
        shape, dtype=dtype, device=_reference_device()
    )

    gems_op = _resolve("_efficientzerotensor")
    res_out = gems_op(shape, dtype=dtype, device=flag_gems.device)

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)


# aten::_efficientzerotensor.out writes zeros into the supplied ``out`` buffer
# and returns that same object (alias semantics). The buffer is pre-filled with
# range-generated garbage via the shared value-range helper.
@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _OUT_RANGE_PARAMS)
def test__efficientzerotensor_out_range(shape, dtype, value_range):
    ref_device = _reference_device()
    garbage = tu.make_input(dtype, shape, value_range)
    ref_buf = garbage.clone().to(ref_device)
    act_buf = garbage.clone()

    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_buf)
    assert ref_out is ref_buf

    gems_op = _resolve("_efficientzerotensor_out")
    res_out = gems_op(shape, out=act_buf)
    assert res_out is act_buf

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device.type == torch.device(flag_gems.device).type
    utils.gems_assert_equal(act_buf, ref_buf)


# Same overload, but every buffer starts from a fixed non-zero sentinel: this
# guarantees a candidate that silently skips the write is caught for every
# dtype, independently of the range framework.
@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_out_overwrites(shape, dtype):
    ref_device = _reference_device()
    ref_buf = torch.full(shape, 1, dtype=dtype, device=ref_device)
    act_buf = torch.full(shape, 1, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_buf)
    assert ref_out is ref_buf

    gems_op = _resolve("_efficientzerotensor_out")
    res_out = gems_op(shape, out=act_buf)
    assert res_out is act_buf

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    utils.gems_assert_equal(act_buf, ref_buf)


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_negative_size():
    # A negative dimension is invalid; the aten reference raises RuntimeError
    # and the candidate must reject it too rather than silently truncating.
    with pytest.raises(RuntimeError):
        torch.ops.aten._efficientzerotensor(
            (-1,), dtype=torch.float32, device=flag_gems.device
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve("_efficientzerotensor")(
            (-1,), dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_non_integer_size():
    # Size elements must be integers; 2.5 cannot match any aten schema.
    with pytest.raises(RuntimeError):
        torch.ops.aten._efficientzerotensor(
            (2.5,), dtype=torch.float32, device=flag_gems.device
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _resolve("_efficientzerotensor")(
            (2.5,), dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_non_strided_layout():
    # Only the strided layout is supported; aten has no sparse kernel and the
    # candidate must reject the request as well.
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._efficientzerotensor(
            (2, 3),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _resolve("_efficientzerotensor")(
            (2, 3),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
