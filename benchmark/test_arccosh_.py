import pytest

import flag_gems

from . import base, consts


@pytest.mark.arccosh_
def test_arccosh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccosh_",
        torch_op=lambda a: a.arccosh_(),
        gems_op=flag_gems.arccosh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
