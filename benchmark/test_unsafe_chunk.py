import pytest
import torch

import flag_gems

from . import base, consts, utils


def _unsafe_chunk_case_fn(shape, dtype):
    # Generate different chunks values for different shapes
    del dtype
    for chunks in [2, 3, 4, 5]:
        # Skip invalid combinations
        dim_size = shape[0]
        if chunks > dim_size:
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"chunks": chunks, "dim": 0},
            builder_args=(shape, chunks),
        )


def _unsafe_chunk_build_inputs_fn(plan, dtype, device):
    shape, chunks = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, chunks, plan.params["dim"]


@pytest.mark.unsafe_chunk
def test_unsafe_chunk():
    bench = base.GenericBenchmark(
        op_name="unsafe_chunk",
        case_fn=_unsafe_chunk_case_fn,
        build_inputs_fn=_unsafe_chunk_build_inputs_fn,
        torch_op=torch.unsafe_chunk,
        gems_op=flag_gems.unsafe_chunk,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
