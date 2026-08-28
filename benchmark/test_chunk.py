import pytest
import torch

import flag_gems

from . import base


def chunk_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Default: split into 3 chunks along dim 0
    yield inp, {"chunks": 3, "dim": 0}


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"chunks": 3, "dim": 0},
        builder_args=(shape, 0),
    )


@pytest.mark.chunk
def test_chunk():
    bench = base.GenericBenchmark(
        op_name="chunk",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(chunk_input_fn),
        torch_op=torch.chunk,
        gems_op=flag_gems.chunk,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
