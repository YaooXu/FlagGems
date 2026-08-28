import pytest
import torch

import flag_gems

from . import base, consts


def _resize_as_case_fn(shape, dtype):
    del dtype
    numel = 1
    for size in shape:
        numel *= size
    target_shape = (numel,)
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "template": target_shape},
        builder_args=(shape, target_shape),
    )


def _resize_as_build_inputs_fn(plan, dtype, device):
    shape, target_shape = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    template = torch.randn(target_shape, dtype=dtype, device=device)
    return inp, template


@pytest.mark.resize_as
def test_resize_as():
    bench = base.GenericBenchmark(
        case_fn=_resize_as_case_fn,
        build_inputs_fn=_resize_as_build_inputs_fn,
        op_name="resize_as",
        torch_op=torch.Tensor.resize_as,
        gems_op=flag_gems.resize_as,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.resize_as_
def test_resize_as_():
    bench = base.GenericBenchmark(
        case_fn=_resize_as_case_fn,
        build_inputs_fn=_resize_as_build_inputs_fn,
        op_name="resize_as_",
        torch_op=torch.Tensor.resize_as_,
        gems_op=flag_gems.resize_as_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )

    bench.run()
