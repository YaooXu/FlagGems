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

# chain_matmul multiplies a sequence of 2-D matrices with an optimized
# parenthesization. Each entry is a chain of (rows, cols) shapes. The default
# shape set contains 1-D tensors, which chain_matmul rejects, so this benchmark
# uses its own performance-relevant 2-D chains.
CHAIN_SHAPES = [
    [(4, 8), (8, 16), (16, 4)],
    [(64, 128), (128, 256), (256, 256), (256, 64)],
    [(128, 256), (256, 512), (512, 256), (256, 128)],
    [(256, 1024), (1024, 1024), (1024, 256)],
    [(512, 4096), (4096, 4096), (4096, 512)],
    [(1024, 4096), (4096, 4096), (4096, 1024)],
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"matrices": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    matrices = [utils.generate_tensor_input(s, dtype, device) for s in shape]
    return (matrices,)


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    matrices = [utils.generate_tensor_input(s, dtype, device) for s in shape]
    out = torch.empty((shape[0][0], shape[-1][1]), dtype=dtype, device=device)
    return matrices, {"out": out}


class ChainMatmulBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to valid 2-D matrix chains."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = CHAIN_SHAPES


@pytest.mark.chain_matmul
def test_chain_matmul():
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.chain_matmul,
        gems_op=getattr(flag_gems, "chain_matmul", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.chain_matmul_out
def test_chain_matmul_out():
    bench = ChainMatmulBenchmark(
        op_name="chain_matmul_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.chain_matmul.out,
        gems_op=getattr(flag_gems, "chain_matmul", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
