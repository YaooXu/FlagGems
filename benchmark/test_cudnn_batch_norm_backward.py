import pytest
import torch

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    # TODO: add new metric

    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def batchnorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.randn((C,), dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled


def cudnn_batch_norm_backward_input_fn(shape, dtype, device):
    for forward_args in batchnorm_input_fn(shape, dtype, device):
        (
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            _,
            eps,
            _,
        ) = forward_args

        grad_output = torch.randn_like(inp)
        channels = weight.shape[0] if weight is not None else inp.shape[1]
        if weight is None:
            weight = torch.ones(channels, dtype=dtype, device=device)
        if bias is None:
            bias = torch.ones(channels, dtype=dtype, device=device)

        inp_f32 = inp.to(torch.float32)
        weight_f32 = weight.to(torch.float32)
        bias_f32 = bias.to(torch.float32)
        _, save_mean, save_var, reserve = torch.ops.aten.cudnn_batch_norm(
            inp_f32, weight_f32, bias_f32, None, None, training, eps, False
        )
        yield (
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean.to(dtype),
            save_var.to(dtype),
            eps,
            reserve.to(dtype),
        )


def cudnn_batch_norm_backward_case_fn(shape, dtype):
    del dtype
    channels = shape[1]
    variants = [False]
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        variants.append(True)
    for input_index, with_running_stats in enumerate(variants):
        yield base.BenchmarkCasePlan(
            shape={
                "input": shape,
                "grad_output": shape,
                "weight": (channels,),
                "running_mean": (channels,) if with_running_stats else None,
                "running_var": (channels,) if with_running_stats else None,
                "save_mean": (channels,),
                "save_var": (channels,),
                "reserve": "backend_generated",
            },
            params={"eps": 1e-5, "with_running_stats": with_running_stats},
            builder_args=(shape, input_index),
        )


@pytest.mark.cudnn_batch_norm_backward
def test_cudnn_batch_norm_backward():
    bench = NormBenchmark(
        case_fn=cudnn_batch_norm_backward_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            cudnn_batch_norm_backward_input_fn
        ),
        op_name="cudnn_batch_norm_backward",
        torch_op=torch.ops.aten.cudnn_batch_norm_backward,
        gems_op=flag_gems.cudnn_batch_norm_backward,
        # cuDNN cudnn_batch_norm_backward only supports float32,
        # so we hardcode float32 instead of using consts.FLOAT_DTYPES.
        dtypes=[torch.float32],
    )
    bench.run()
