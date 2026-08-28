import pytest
import torch

import flag_gems

from . import base, consts, utils


def _logit_backward_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"grad_output": shape, "input": shape},
        params={"eps": 1e-6},
        builder_args=(shape,),
    )


def _logit_backward_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    eps = plan.params["eps"]
    inp = torch.rand(shape, dtype=dtype, device=device)
    grad_output = torch.ones(shape, dtype=dtype, device=device)
    return grad_output, inp, eps


@pytest.mark.logit_backward
def test_logit_backward():
    bench = base.GenericBenchmark(
        op_name="logit_backward",
        case_fn=_logit_backward_case_fn,
        build_inputs_fn=_logit_backward_build_inputs_fn,
        torch_op=torch.ops.aten.logit_backward,
        gems_op=flag_gems.logit_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
