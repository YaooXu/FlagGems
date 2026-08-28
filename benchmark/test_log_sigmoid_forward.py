import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.log_sigmoid_forward
def test_log_sigmoid_forward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log_sigmoid_forward",
        torch_op=torch.ops.aten.log_sigmoid_forward,
        gems_op=flag_gems.log_sigmoid_forward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
