import pytest
import torch

import flag_gems

from . import base

# Square matrices from 2x2 to 256x256 covering small to medium-large use cases
CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


class CholeskyInverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_INVERSE_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"L": list(shape)},
                    params={},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        n = shape[-1]
        # Create positive-definite matrix and get its Cholesky factor
        B = torch.randn(shape, dtype=case.dtype, device=self.device)
        A = (
            B @ B.transpose(-2, -1)
            + torch.eye(n, dtype=case.dtype, device=self.device) * n
        )
        L = torch.linalg.cholesky(A)
        return (L,)


@pytest.mark.cholesky_inverse
def test_cholesky_inverse():
    bench = CholeskyInverseBenchmark(
        op_name="cholesky_inverse",
        torch_op=torch.cholesky_inverse,
        gems_op=flag_gems.cholesky_inverse,
        # cholesky_inverse only supports float32/float64
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
