import pytest
import torch

import flag_gems

from . import base, consts

# Shapes covering 2D, 3D, and 4D for benchmarking transpose.
# transpose.int is a zero-copy view op, so the benchmark measures
# dispatch + as_strided overhead rather than data movement.
TRANSPOSE_SHAPES = [
    (64, 64),
    (256, 512),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 64),
    (8, 16, 32, 64),
]


class TransposeBenchmark(base.Benchmark):
    """Benchmark for aten::transpose.int (zero-copy view operation)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = TRANSPOSE_SHAPES

    def get_input_iter(self, dtype):
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            # Swap first and last dimensions for every shape.
            dim0 = 0
            dim1 = len(shape) - 1
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"dim0": dim0, "dim1": dim1},
                    builder_args=(shape, dim0, dim1),
                ),
            )

    def build_inputs(self, case):
        shape, dim0, dim1 = case.builder_args[0].builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, dim0, dim1


@pytest.mark.transpose
def test_transpose():
    bench = TransposeBenchmark(
        op_name="transpose",
        torch_op=torch.ops.aten.transpose.int,
        gems_op=flag_gems.transpose,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
