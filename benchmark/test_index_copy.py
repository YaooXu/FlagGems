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


class IndexCopyBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [(1, 2), (4096, 256), (200, 40999, 3)]

    def get_gbps(self, bench_fn_args, latency):
        index = bench_fn_args[2]
        src = bench_fn_args[3]
        io_amount = sum([utils.size_in_bytes(item) for item in [index, src, src]])
        return io_amount * 1e-9 / (latency * 1e-3)


def _tensor_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


def _case_fn(shape, dtype):
    del dtype
    dim = 0 if len(shape) == 1 else 1
    index_len = max(shape[dim] // 2, 1)
    src_shape = list(shape)
    src_shape[dim] = index_len
    yield base.BenchmarkCasePlan(
        shape={
            "input": shape,
            "index": (index_len,),
            "source": tuple(src_shape),
        },
        params={"dim": dim},
        builder_args=(shape, 0),
    )


@pytest.mark.index_copy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_copy():
    bench = IndexCopyBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_tensor_input_fn),
        op_name="index_copy",
        torch_op=torch.index_copy,
        gems_op=flag_gems.index_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.index_copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_copy_():
    bench = IndexCopyBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(_tensor_input_fn),
        op_name="index_copy_",
        torch_op=torch.Tensor.index_copy_,
        gems_op=flag_gems.index_copy_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
