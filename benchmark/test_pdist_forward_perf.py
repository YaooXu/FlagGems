import pytest
import torch

import flag_gems

from . import base

# Covers representative (N, M) pairs exercising different grid sizes and BLOCK_M widths.
PDIST_FORWARD_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class PdistForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_FORWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, 2.0

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            n, m = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"x": (n, m)},
                    params={"p": 2.0},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        return x, plan.params["p"]


@pytest.mark.pdist_forward
def test_pdist_forward():
    bench = PdistForwardBenchmark(
        op_name="pdist_forward",
        torch_op=torch.ops.aten._pdist_forward,
        gems_op=flag_gems._pdist_forward,
        # _pdist_forward only supports float32 in the reference implementation
        dtypes=[torch.float32],
    )
    bench.run()
