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


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Volumetric feature maps with symmetric and asymmetric padding.
        self.shapes = [
            (2, 3, 8, 16, 16, 1, 1, 1, 1, 1, 1),
            (2, 8, 16, 32, 32, 2, 1, 1, 2, 3, 1),
            (1, 16, 32, 32, 32, 1, 1, 1, 1, 1, 1),
        ]
        self.shape_desc = "N, C, D, H, W, PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM, PAD_FRONT, PAD_BACK"

    def get_case_iter(self, dtype):
        for ordinal, dims in enumerate(self.shapes):
            n, c, d, h, w, pl, pr, pt, pb, pf, pk = dims
            padding = (pl, pr, pt, pb, pf, pk)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "grad_output": (
                            n,
                            c,
                            d + pf + pk,
                            h + pt + pb,
                            w + pl + pr,
                        )
                    },
                    params={"inp_shape": [n, c, d, h, w], "padding": list(padding)},
                    builder_args=(n, c, d, h, w, padding),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, c, d, h, w, padding = plan.builder_args
        inp = torch.randn((n, c, d, h, w), dtype=case.dtype, device=self.device)
        pl, pr, pt, pb, pf, pk = padding
        grad_output = torch.randn(
            (n, c, d + pf + pk, h + pt + pb, w + pl + pr),
            dtype=case.dtype,
            device=self.device,
        )
        return grad_output, inp, padding


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        gems_op=flag_gems.replication_pad3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
