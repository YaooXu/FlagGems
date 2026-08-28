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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, utils


class MaskedScaleBenchmark(base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if flag_gems.vendor_name == "cambricon":
                # Cambricon torch.randint currently does not support uint8 generation.
                mask = torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
                    self.device
                )
            else:
                mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=self.device)
            scale = 2.0
            yield inp, mask, scale

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape, "mask": shape},
                    params={"scale": 2.0, "mask_dtype": "torch.uint8"},
                    builder_args=(shape, 0),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        if flag_gems.vendor_name == "cambricon":
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device="cpu").to(
                self.device
            )
        else:
            mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=self.device)
        scale = 2.0
        return inp, mask, scale


# _masked_scale only supports float32 on most backends.
# CUDA reference does not support float16/bf16 for this private op.
FLOAT_DTYPES = [torch.float32]


@pytest.mark.masked_scale
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_masked_scale(dtype):
    bench = MaskedScaleBenchmark(
        op_name="masked_scale",
        torch_op=torch.ops.aten._masked_scale,
        gems_op=flag_gems._masked_scale,
        dtypes=[dtype],
    )
    bench.run()
