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

from . import base, consts


class ChannelShuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative shapes covering small to medium NCHW inputs for channel shuffle
        self.shapes = [
            ((1, 4, 2, 2), 2),
            ((2, 8, 4, 4), 4),
            ((4, 16, 8, 8), 4),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape, groups in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield x, groups

    def get_case_iter(self, dtype):
        for ordinal, shape_config in enumerate(self.shapes):
            shape, groups = shape_config
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"groups": groups},
                    builder_args=(shape_config, 0),
                ),
            )

    def build_inputs(self, case):
        shape, groups = case.builder_args[0].builder_args[0]
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        return x, groups


@pytest.mark.channel_shuffle
def test_channel_shuffle():
    bench = ChannelShuffleBenchmark(
        op_name="channel_shuffle",
        torch_op=torch.ops.aten.channel_shuffle,
        gems_op=flag_gems.channel_shuffle,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
