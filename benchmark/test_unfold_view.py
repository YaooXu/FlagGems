import pytest
import torch

import flag_gems

from . import base


def _unfold_case_fn(shape, dtype):
    del dtype
    # Unfold along dim 0 with size=4, step=2
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dimension": 0, "size": 4, "step": 2},
        builder_args=(shape,),
    )


def _unfold_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, {
        "dimension": plan.params["dimension"],
        "size": plan.params["size"],
        "step": plan.params["step"],
    }


@pytest.mark.unfold
def test_unfold_view():
    bench = base.GenericBenchmark(
        case_fn=_unfold_case_fn,
        build_inputs_fn=_unfold_build_inputs_fn,
        op_name="unfold",
        torch_op=torch.Tensor.unfold,
        gems_op=flag_gems.unfold,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
