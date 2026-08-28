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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.new_full
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "xdtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "fill_value", [3.1415926, 2, False, float("inf"), float("nan")]
)
def test_new_full(shape, dtype, xdtype, fill_value):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    special_value = isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    )
    implicit_supported = not special_value or dtype in utils.ALL_FLOAT_DTYPES
    explicit_supported = not special_value or xdtype in utils.ALL_FLOAT_DTYPES
    if not implicit_supported and not explicit_supported:
        pytest.skip("inf/nan fill values require a floating output dtype")

    gems_op = flag_gems.testing.resolve_gems_op("new_full", flag_gems.new_full)

    if implicit_supported:
        ref_out = ref_inp.new_full(shape, fill_value)
        res_out = gems_op(inp, shape, fill_value)
        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)

    if explicit_supported:
        ref_out = ref_inp.new_full(shape, fill_value, dtype=xdtype)
        res_out = gems_op(inp, shape, fill_value, dtype=xdtype)
        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
