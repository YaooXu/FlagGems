import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.ixor
def test_ixor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ixor",
        torch_op=torch.ops.aten.__ixor__,
        gems_op=flag_gems.xor_,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
