import pytest

import flag_gems

from . import base, consts


@pytest.mark.lcm_
def test_lcm_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm_",
        torch_op=lambda a, b: a.lcm_(b),
        gems_op=flag_gems.lcm_,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
