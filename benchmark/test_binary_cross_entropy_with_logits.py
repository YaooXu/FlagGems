import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    yield inp, target


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "target": shape},
        builder_args=(shape, 0),
    )


@pytest.mark.binary_cross_entropy_with_logits
def test_binary_cross_entropy_with_logits():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn),
        op_name="binary_cross_entropy_with_logits",
        torch_op=torch.ops.aten.binary_cross_entropy_with_logits,
        gems_op=flag_gems.binary_cross_entropy_with_logits,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
