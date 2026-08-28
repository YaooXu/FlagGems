import pytest
import torch

import flag_gems

from . import base


def _alias_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _alias_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, {}


@pytest.mark.alias
def test_alias():
    bench = base.GenericBenchmark(
        op_name="alias",
        case_fn=_alias_case_fn,
        build_inputs_fn=_alias_build_inputs_fn,
        torch_op=torch.ops.aten.alias.default,
        gems_op=flag_gems.alias,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
