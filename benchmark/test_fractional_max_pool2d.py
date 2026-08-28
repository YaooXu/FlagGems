import pytest
import torch

import flag_gems

from . import base, consts, utils


class FractionalMaxPool2dBenchmark(base.GenericBenchmark):
    pass


def fractional_max_pool2d_case_fn(shape, dtype):
    del dtype
    output_sizes = [(shape[2] // 2, shape[3] // 2)]
    if (
        base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        and shape[-2] > 5
        and shape[-1] > 5
    ):
        output_sizes.append((shape[2] // 4, shape[3] // 4))
    for input_index, output_size in enumerate(output_sizes):
        yield base.BenchmarkCasePlan(
            shape={"input": shape, "output": shape[:2] + output_size},
            params={"kernel_size": 2, "output_size": output_size},
            builder_args=(shape, input_index),
        )


def fractional_max_pool2d_backward_case_fn(shape, dtype):
    del dtype
    output_size = (shape[2] // 2, shape[3] // 2)
    yield base.BenchmarkCasePlan(
        shape={
            "grad_output": shape[:2] + output_size,
            "input": shape,
            "indices": shape[:2] + output_size,
        },
        params={"kernel_size": 2, "output_size": output_size},
        builder_args=(shape, 0),
    )


def fractional_max_pool2d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)}
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": 2,
                "output_size": (shape[2] // 4, shape[3] // 4),
            }


@pytest.mark.fractional_max_pool2d
def test_fractional_max_pool2d():
    bench = FractionalMaxPool2dBenchmark(
        case_fn=fractional_max_pool2d_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            fractional_max_pool2d_input_fn
        ),
        op_name="fractional_max_pool2d",
        torch_op=torch.nn.functional.fractional_max_pool2d,
        gems_op=flag_gems.fractional_max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def fractional_max_pool2d_backward_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    inp.requires_grad_(True)
    output_size = (shape[2] // 2, shape[3] // 2)
    output, indices = flag_gems.fractional_max_pool2d(
        inp, kernel_size=2, output_size=output_size
    )
    grad_output = torch.randn_like(output)
    yield grad_output, inp, {
        "kernel_size": 2,
        "output_size": output_size,
        "indices": indices,
    }


def torch_fractional_max_pool2d_backward_wrapper(grad_output, input, **kwargs):
    return torch.ops.aten.fractional_max_pool2d_backward(
        grad_output,
        input,
        kwargs["kernel_size"],
        kwargs["output_size"],
        kwargs["indices"],
    )


@pytest.mark.fractional_max_pool2d_backward
def test_fractional_max_pool2d_backward():
    bench = FractionalMaxPool2dBenchmark(
        case_fn=fractional_max_pool2d_backward_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            fractional_max_pool2d_backward_input_fn
        ),
        op_name="fractional_max_pool2d_backward",
        torch_op=torch_fractional_max_pool2d_backward_wrapper,
        gems_op=flag_gems.fractional_max_pool2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
