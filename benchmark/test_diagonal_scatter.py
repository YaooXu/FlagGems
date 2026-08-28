import pytest
import torch

import flag_gems

from . import base, consts

# Shapes for diagonal_scatter benchmark
DIAGONAL_SCATTER_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (64, 128),
    (128, 256),
    (256, 512),
    (32, 64, 64),
    (64, 128, 128),
    (16, 32, 32, 32),
]


class DiagonalScatterBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_SCATTER_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            diagonal_shape = shape[:-2] + (min(shape[-2:]),)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape, "src": diagonal_shape},
                    params={"offset": 0, "dim1": -2, "dim2": -1},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        diag = torch.diagonal(inp, 0, -2, -1)
        src = torch.randn(diag.shape, dtype=case.dtype, device=self.device)
        return inp, src, 0, -2, -1


@pytest.mark.diagonal_scatter
def test_diagonal_scatter():
    bench = DiagonalScatterBenchmark(
        op_name="diagonal_scatter",
        torch_op=torch.diagonal_scatter,
        gems_op=flag_gems.diagonal_scatter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
