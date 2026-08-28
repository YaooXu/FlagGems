import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.special_digamma
def test_special_digamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_digamma",
        torch_op=torch.special.digamma,
        gems_op=flag_gems.special_digamma,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
