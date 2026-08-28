import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.special_erfc
def test_special_erfc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_erfc",
        torch_op=torch.ops.aten.special_erfc,
        gems_op=flag_gems.special_erfc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.erfc
def test_erfc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfc",
        torch_op=torch.ops.aten.erfc,
        gems_op=flag_gems.erfc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.erfc_
def test_erfc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfc_",
        torch_op=torch.ops.aten.erfc_,
        gems_op=flag_gems.erfc_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
