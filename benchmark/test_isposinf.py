import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.isposinf
def test_isposinf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isposinf",
        torch_op=torch.isposinf,
        gems_op=flag_gems.isposinf,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
