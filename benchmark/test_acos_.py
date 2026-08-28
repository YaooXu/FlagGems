import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.acos_
def test_acos_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acos_",
        torch_op=torch.acos_,
        gems_op=flag_gems.acos_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
