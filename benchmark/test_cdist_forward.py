import pytest
import torch

import flag_gems

from . import base

# Shapes for cdist benchmark: (P, M), (R, M) -> (P, R)
# torch.cdist doesn't support float16 on CUDA
CDIST_FORWARD_SHAPES = [
    ((4, 8), (6, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (16, 32)),
    ((32, 64), (32, 64)),
    ((64, 128), (64, 128)),
]


class CdistForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CDIST_FORWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape1, shape2 in self.shapes:
            x1 = torch.randn(*shape1, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(*shape2, dtype=cur_dtype, device=self.device)
            yield x1, x2, 2.0

    def get_case_iter(self, dtype):
        for ordinal, (shape1, shape2) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"x1": list(shape1), "x2": list(shape2)},
                    params={"p": 2.0},
                    builder_args=(shape1, shape2),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape1, shape2 = plan.builder_args
        x1 = torch.randn(*shape1, dtype=case.dtype, device=self.device)
        x2 = torch.randn(*shape2, dtype=case.dtype, device=self.device)
        return x1, x2, 2.0

    def get_tflops(self, op, *args, **kwargs):
        x1, x2, _ = args
        # FLOPs = 2 * P * R * M (for L2 distance computation)
        return 2 * x1.shape[-2] * x2.shape[-2] * x1.shape[-1]


@pytest.mark.cdist_forward
def test_cdist_forward():
    bench = CdistForwardBenchmark(
        op_name="cdist_forward",
        torch_op=torch.cdist,
        gems_op=flag_gems._cdist_forward,
        # torch.cdist doesn't support float16 on CUDA; only float32 is numerically stable
        dtypes=[torch.float32],
    )
    bench.run()
