import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.lcm
def test_lcm():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm",
        torch_op=torch.lcm,
        gems_op=flag_gems.lcm,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
