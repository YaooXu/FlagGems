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
from flag_gems.utils import shape_utils

from . import base, consts


class ScatterReduceBenchmark(base.GenericBenchmark2DOnly):
    def get_gbps(self, args, latency):
        inp, _, index, src = args[:4]
        io_amount = sum(
            shape_utils.size_in_bytes(item)
            for item in (inp, index, src, inp)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


def _input_fn_factory(reduce):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        size_dim = shape[dim]
        index = torch.randint(0, size_dim, shape, dtype=torch.long, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)
        yield inp, dim, index, src, {"reduce": reduce}

    return inner


def _case_fn_factory(reduce):
    def inner(shape, dtype):
        del dtype
        yield base.BenchmarkCasePlan(
            shape={"self": shape, "index": shape, "src": shape},
            params={"dim": -1, "reduce": reduce},
            builder_args=(shape, 0),
        )

    return inner


@pytest.mark.scatter_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_sum():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        gems_op=flag_gems.scatter_reduce_,
        case_fn=_case_fn_factory("sum"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            _input_fn_factory("sum")
        ),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_amax():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        gems_op=flag_gems.scatter_reduce_,
        case_fn=_case_fn_factory("amax"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            _input_fn_factory("amax")
        ),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_amin():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        gems_op=flag_gems.scatter_reduce_,
        case_fn=_case_fn_factory("amin"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            _input_fn_factory("amin")
        ),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_mean():
    bench = ScatterReduceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        gems_op=flag_gems.scatter_reduce_,
        case_fn=_case_fn_factory("mean"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            _input_fn_factory("mean")
        ),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
