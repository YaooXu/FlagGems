import pytest
import torch

import flag_gems

from . import base, consts, utils


def _case_fn(shape, dtype):
    del dtype
    # Unflatten dim 0 into different factorizations
    dim = 0
    dim_size = shape[dim]

    # Find valid factorizations
    factors = []
    for f in range(2, min(dim_size + 1, 17)):
        if dim_size % f == 0:
            factors.append(f)

    if not factors:
        factors = [1]

    for factor in factors[:4]:
        sizes = (factor, dim_size // factor)
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": dim, "sizes": list(sizes)},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {
        "dim": plan.params["dim"],
        "sizes": tuple(plan.params["sizes"]),
    }


@pytest.mark.unflatten
def test_unflatten():
    bench = base.GenericBenchmark(
        op_name="unflatten",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.unflatten,
        gems_op=flag_gems.unflatten,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
