import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.acos_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acos_(ref_inp)
    gems_op = flag_gems.testing.resolve_gems_op("acos_", flag_gems.acos_)
    res_out = gems_op(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    # Verify the mutated input matches the returned result
    assert ref_inp is ref_out
    utils.gems_assert_close(inp, ref_inp, dtype, True)
