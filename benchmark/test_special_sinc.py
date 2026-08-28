import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.special_sinc
def test_special_sinc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_sinc",
        torch_op=torch.special.sinc,
        gems_op=flag_gems.special_sinc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
