import pytest
import torch

import flag_gems

from . import base


@pytest.mark.special_bessel_y0
def test_special_bessel_y0():
    class BesselY0Benchmark(base.UnaryPointwiseBenchmark):
        def build_inputs(self, case):
            plan = case.builder_args[0]
            shape = plan.builder_args[0]
            if case.dtype == torch.float64:
                inp = torch.randn(shape, dtype=torch.float64, device=self.device)
            else:
                inp = base.generate_tensor_input(shape, case.dtype, self.device)
            return (inp,)

    bench = BesselY0Benchmark(
        op_name="special_bessel_y0",
        torch_op=torch.special.bessel_y0,
        gems_op=flag_gems.special_bessel_y0,
        # torch.special.bessel_y0 supports float32 and float64
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
