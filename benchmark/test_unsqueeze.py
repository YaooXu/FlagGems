import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    # Use dim=0 for 1D, dim=1 for 2D+
    if len(shape) == 1:
        yield inp, 0
    elif len(shape) >= 2:
        yield inp, 1


def _case_fn(shape, dtype):
    del dtype
    dim = 0 if len(shape) == 1 else 1
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dim": dim},
        builder_args=(shape, 0),
    )


@pytest.mark.unsqueeze
def test_unsqueeze():
    bench = base.GenericBenchmark(
        op_name="unsqueeze",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn),
        torch_op=torch.unsqueeze,
        gems_op=flag_gems.unsqueeze,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unsqueeze_
def test_unsqueeze_():
    bench = base.GenericBenchmark(
        op_name="unsqueeze_",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_input_fn),
        torch_op=torch.Tensor.unsqueeze_,
        gems_op=flag_gems.unsqueeze_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
