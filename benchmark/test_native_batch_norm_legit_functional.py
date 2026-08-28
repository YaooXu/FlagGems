import pytest
import torch

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def _norm_case_fn(shape, dtype):
    del dtype
    C = shape[1]
    yield base.BenchmarkCasePlan(
        shape={
            "input": list(shape),
            "weight": [C],
            "bias": [C],
            "running_mean": [C],
            "running_var": [C],
        },
        params={"training": True, "momentum": 0.1, "eps": 1e-5},
        builder_args=(shape,),
    )


def _norm_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)
    running_mean = torch.zeros(C, dtype=dtype, device=device)
    running_var = torch.ones(C, dtype=dtype, device=device)
    return inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5


@pytest.mark.native_batch_norm_legit_functional
def test_native_batch_norm_legit_functional():
    bench = NormBenchmark(
        case_fn=_norm_case_fn,
        build_inputs_fn=_norm_build_inputs_fn,
        op_name="native_batch_norm_legit_functional",
        torch_op=torch.ops.aten._native_batch_norm_legit_functional.default,
        gems_op=flag_gems._native_batch_norm_legit_functional,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
