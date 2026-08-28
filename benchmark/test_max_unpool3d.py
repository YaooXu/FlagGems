import pytest
import torch

import flag_gems

from . import base, consts

# Benchmark shapes for max_unpool3d
MAX_UNPOOL3D_BENCH_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 16, 16, 16),
    (4, 4, 8, 8, 8),
]


class MaxUnpool3dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MAX_UNPOOL3D_BENCH_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"kernel_size": 2, "stride": 2, "padding": 0},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        # Generate pooled input and indices from max_pool3d
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        output, indices = pool(x)
        # Output size should be the original input shape
        # kernel_size, stride, padding, output_size(D,H,W)
        return output, indices, 2, 2, 0, shape[2:]


@pytest.mark.max_unpool3d
def test_max_unpool3d():
    bench = MaxUnpool3dBenchmark(
        op_name="max_unpool3d",
        torch_op=torch.nn.functional.max_unpool3d,
        gems_op=flag_gems.max_unpool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
