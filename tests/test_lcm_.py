import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.inplace
@pytest.mark.lcm_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lcm_(shape, dtype):
    inp1 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.lcm_(ref_inp2)
    gems_op = flag_gems.testing.resolve_gems_op("lcm_", flag_gems.lcm_)
    res_out = gems_op(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)
