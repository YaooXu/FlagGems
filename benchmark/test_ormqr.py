import pytest
import torch

import flag_gems

from . import base


def _ormqr_input_fn(shape, dtype, device):
    m, n = shape
    k = min(m, n)
    # Generate valid Householder reflectors via QR decomposition
    a = torch.randn(m, k, dtype=dtype, device=device)
    input_tensor, tau = torch.geqrf(a)
    other = torch.randn(m, n, dtype=dtype, device=device)
    yield input_tensor, tau, other


# ormqr benchmark
# ormqr applies Householder reflectors which have inherent sequential dependency.
# The fused Triton kernel excels on small-to-medium matrices where the entire
# active region fits in GPU SRAM (<=128 in both dimensions).
class OrmqrBenchmark(base.GenericBenchmark2DOnly):
    # Override default shapes to include representative sizes for Householder application:
    # small matrices (fused kernel path) and medium/large matrices (tiled path)
    DEFAULT_SHAPES = [
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 96),
        (128, 128),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (1024, 65536),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        # ormqr: multiply Q (m x m or n x n) with C (m x n)
        # Flops: 2 * m * n * min(m, n) for the matrix multiplication
        m, n = args[2].shape
        k = args[0].shape[-1]  # k = min(m, n) for ormqr
        return 2 * m * n * k

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            m, n = shape
            k = min(m, n)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": (m, k), "tau": (k,), "other": (m, n)},
                    params={},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape = plan.builder_args[0]
        m, n = shape
        k = min(m, n)
        # Generate valid Householder reflectors via QR decomposition
        a = torch.randn(m, k, dtype=case.dtype, device=self.device)
        input_tensor, tau = torch.geqrf(a)
        other = torch.randn(m, n, dtype=case.dtype, device=self.device)
        return input_tensor, tau, other


@pytest.mark.ormqr
def test_ormqr():
    bench = OrmqrBenchmark(
        input_fn=_ormqr_input_fn,
        op_name="ormqr",
        torch_op=torch.ormqr,
        gems_op=flag_gems.ormqr,
        # ormqr only supports float32 and float64 (LAPACK limitation, no half/bfloat16)
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
