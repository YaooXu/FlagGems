import pytest
import torch

import flag_gems

from . import base

# Shapes for linalg_svdvals benchmark
SVD_BENCHMARK_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
]


class SvdBenchmark(base.GenericBenchmark2DOnly):
    """
    Benchmark for linalg_svdvals
    """

    def set_more_shapes(self):
        return SVD_BENCHMARK_SHAPES


def svd_input_fn(shape, cur_dtype, device):
    del cur_dtype
    m, n = shape
    inp = torch.randn([m, n], dtype=torch.float32, device=device)
    yield inp,


def svd_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        builder_args=(shape, 0),
    )


@pytest.mark.linalg_svdvals
def test_linalg_svdvals():
    bench = SvdBenchmark(
        case_fn=svd_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(svd_input_fn),
        op_name="linalg_svdvals",
        torch_op=torch.linalg.svdvals,
        gems_op=flag_gems.linalg_svdvals,
        # Only float32 for SVD on CUDA (PyTorch limitation)
        dtypes=[torch.float32],
    )
    bench.run()
