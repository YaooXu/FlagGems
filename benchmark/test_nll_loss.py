# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base, consts, utils


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


def nll_loss_case_fn(shape, dtype):
    del dtype
    target_shape = list(shape)
    del target_shape[1]
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "target": target_shape},
        builder_args=(shape, False),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={
                "input": shape,
                "target": target_shape,
                "weight": (shape[1],),
            },
            params={"ignore_index": 1, "reduction": "none"},
            builder_args=(shape, True),
        )


def materialize_nll_loss_case(plan, dtype, device):
    shape, with_weight = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    if not with_weight:
        return inp, target
    weight = torch.randn(shape[1], dtype=dtype, device=device)
    return inp, target, {"weight": weight, **plan.params}


@pytest.mark.nll_loss_forward
def test_nll_loss_forward():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss_forward",
        case_fn=nll_loss_case_fn,
        build_inputs_fn=materialize_nll_loss_case,
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def nll_loss_backward_case_fn(shape, dtype):
    del dtype
    target_shape = list(shape)
    del target_shape[1]
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "target": target_shape},
        params={"reduction": 1},
        builder_args=(shape, False, 1),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={
                "self": shape,
                "target": target_shape,
                "weight": (shape[1],),
            },
            params={"reduction": 0, "ignore_index": 1},
            builder_args=(shape, True, 0),
        )


def materialize_nll_loss_backward_case(plan, dtype, device):
    shape, with_weight, reduction = plan.builder_args
    target_shape = list(shape)
    del target_shape[1]
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = torch.randint(0, shape[-1], target_shape, device=device)
    weight = (
        torch.randn(shape[1], dtype=dtype, device=device) if with_weight else None
    )
    ignore_index = plan.params.get("ignore_index", -100)
    valid_target = target != ignore_index
    if weight is None:
        total_weight = valid_target.sum().to(dtype)
    else:
        total_weight = weight[target[valid_target]].sum()
    if reduction == 0:
        grad_output = utils.generate_tensor_input(target_shape, dtype, device)
    else:
        grad_output = utils.generate_tensor_input([], dtype, device)
    return (
        grad_output,
        inp,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    )


@pytest.mark.nll_loss_backward
def test_nll_loss_backward():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss_backward",
        case_fn=nll_loss_backward_case_fn,
        build_inputs_fn=materialize_nll_loss_backward_case,
        torch_op=torch.ops.aten.nll_loss_backward,
        gems_op=flag_gems.nll_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
