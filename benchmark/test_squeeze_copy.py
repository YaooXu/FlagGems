import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.squeeze_copy
def test_squeeze_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="squeeze_copy",
        torch_op=torch.squeeze_copy,
        gems_op=flag_gems.squeeze_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
