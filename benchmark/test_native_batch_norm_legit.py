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
from flag_gems.ops._native_batch_norm_legit import (
    _native_batch_norm_legit_no_stats,
    _native_batch_norm_legit_no_stats_out,
    _native_batch_norm_legit_out,
)

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
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


def _make_common_inputs(shape, dtype, device):
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    return inp, weight, bias


def _default_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"training": True, "momentum": 0.1, "eps": 1e-5},
        builder_args=(shape,),
    )


def _default_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    channels = shape[1]
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    return inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5


def _no_stats_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    yield inp, weight, bias, True, 0.1, 1e-5


def _out_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    yield inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5, outputs


def _no_stats_out_input_fn(shape, dtype, device):
    inp, weight, bias = _make_common_inputs(shape, dtype, device)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    outputs = {
        "out": torch.empty_like(inp),
        "save_mean": torch.empty(channels, dtype=stats_dtype, device=device),
        "save_invstd": torch.empty(channels, dtype=stats_dtype, device=device),
    }
    yield inp, weight, bias, True, 0.1, 1e-5, outputs


def _run_benchmark(op_name, torch_op, gems_op, input_fn):
    bench = NormBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.native_batch_norm_legit
def test_native_batch_norm_legit():
    bench = NormBenchmark(
        op_name="native_batch_norm_legit",
        case_fn=_default_case_fn,
        build_inputs_fn=_default_build_inputs_fn,
        torch_op=torch.ops.aten._native_batch_norm_legit.default,
        gems_op=flag_gems._native_batch_norm_legit,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.native_batch_norm_legit_no_stats
def test_native_batch_norm_legit_no_stats():
    _run_benchmark(
        "native_batch_norm_legit_no_stats",
        torch.ops.aten._native_batch_norm_legit.no_stats,
        _native_batch_norm_legit_no_stats,
        _no_stats_input_fn,
    )


@pytest.mark.native_batch_norm_legit_out
def test_native_batch_norm_legit_out():
    _run_benchmark(
        "native_batch_norm_legit_out",
        torch.ops.aten._native_batch_norm_legit.out,
        _native_batch_norm_legit_out,
        _out_input_fn,
    )


@pytest.mark.native_batch_norm_legit_no_stats_out
def test_native_batch_norm_legit_no_stats_out():
    _run_benchmark(
        "native_batch_norm_legit_no_stats_out",
        torch.ops.aten._native_batch_norm_legit.no_stats_out,
        _native_batch_norm_legit_no_stats_out,
        _no_stats_out_input_fn,
    )
