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

from . import base


class FusedMovingAvgObsFqBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Representative QAT tensor sizes: large per-tensor activations and
        # per-channel conv/linear weight tensors. Small tensors are dominated by
        # fixed kernel-launch overhead and are not a realistic QAT workload.
        return [
            (1 << 26,),  # 64M per-tensor activation
            (1 << 28,),  # 256M per-tensor activation
            (2048, 16384),  # per-channel weights
            (4096, 16384),  # per-channel weights
        ]


def _case_fn(shape, dtype):
    del dtype
    # aten._fused_moving_avg_obs_fq_helper requires a float32 `self`; the qparam
    # channel count is 1 for per-tensor and shape[0] for per-channel.
    per_channel = len(shape) > 1
    n = shape[0] if per_channel else 1
    yield base.BenchmarkCasePlan(
        shape={"inp": list(shape)},
        params={
            "n": n,
            "averaging_const": 0.01,
            "quant_min": 0,
            "quant_max": 255,
            "ch_axis": 0,
            "per_row_fake_quant": per_channel,
            "symmetric_quant": False,
        },
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    n = plan.params["n"]
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    observer_on = torch.tensor(1, dtype=torch.long, device=device)
    fake_quant_on = torch.tensor(1, dtype=torch.long, device=device)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=device)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=device)
    scale = torch.ones((n,), dtype=torch.float32, device=device)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=device)
    return (
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        plan.params["averaging_const"],
        plan.params["quant_min"],
        plan.params["quant_max"],
        plan.params["ch_axis"],
        plan.params["per_row_fake_quant"],
        plan.params["symmetric_quant"],
    )


def _torch_fused_moving_avg_obs_fq_helper(
    inp,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant,
    symmetric_quant,
):
    return torch.ops.aten._fused_moving_avg_obs_fq_helper(
        inp,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row_fake_quant,
        symmetric_quant,
    )


@pytest.mark.fused_moving_avg_obs_fq_helper
def test_fused_moving_avg_obs_fq_helper():
    bench = FusedMovingAvgObsFqBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="fused_moving_avg_obs_fq_helper",
        torch_op=_torch_fused_moving_avg_obs_fq_helper,
        gems_op=flag_gems._fused_moving_avg_obs_fq_helper,
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
