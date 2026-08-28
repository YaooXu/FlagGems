import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.arctan2
def test_arctan2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="arctan2",
        torch_op=torch.arctan2,
        gems_op=flag_gems.arctan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.arctan2_
def test_arctan2_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="arctan2_",
        torch_op=lambda a, b: a.arctan2_(b),
        gems_op=flag_gems.arctan2_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
