import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.prelu_kernel
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_prelu_kernel(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten._prelu_kernel(ref_inp1, ref_inp2)
    gems_op = flag_gems.testing.resolve_gems_op(
        "prelu_kernel", flag_gems._prelu_kernel
    )
    res_out = gems_op(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
