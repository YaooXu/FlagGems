import pytest
import torch

import flag_gems

from . import base


def expand_as_input_fn(shape, dtype, device):
    # Create a tensor with size-1 dims that can be broadcast to shape
    inp_shape = [1 if i == 0 else s for i, s in enumerate(shape)]
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"other": other}


def _expand_as_case_fn(shape, dtype):
    del dtype
    inp_shape = [1 if i == 0 else s for i, s in enumerate(shape)]
    yield base.BenchmarkCasePlan(
        shape={"inp": inp_shape, "other": list(shape)},
        builder_args=(shape, 0),
    )


@pytest.mark.expand_as
def test_expand_as():
    bench = base.GenericBenchmark(
        case_fn=_expand_as_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(expand_as_input_fn),
        op_name="expand_as",
        torch_op=torch.Tensor.expand_as,
        gems_op=flag_gems.expand_as,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
