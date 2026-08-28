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


def empty_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        builder_args=(shape,),
    )


def empty_build_inputs_fn(plan, dtype, device):
    del dtype, device
    # Keep ``size`` as one public ABI argument instead of flattening the shape
    # tuple into multiple positional arguments.
    return (plan.builder_args[0],)


def empty_permuted_case_fn(shape, dtype):
    # Reverse the physical layout so the allocation exercises a non-contiguous
    # memory ordering rather than the plain contiguous one.
    del dtype
    physical_layout = list(reversed(range(len(shape))))
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={"physical_layout": physical_layout},
        builder_args=(shape,),
    )


def empty_permuted_build_inputs_fn(plan, dtype, device):
    del dtype, device
    return plan.builder_args[0], plan.params["physical_layout"]


@pytest.mark.empty_permuted
def test_empty_permuted():
    bench = base.GenericBenchmark(
        op_name="empty_permuted",
        torch_op=torch.empty_permuted,
        gems_op=flag_gems.empty_permuted,
        dtypes=consts.FLOAT_DTYPES,
        case_fn=empty_permuted_case_fn,
        build_inputs_fn=empty_permuted_build_inputs_fn,
    )
    bench.run()


@pytest.mark.empty
def test_empty():
    bench = base.GenericBenchmark(
        op_name="empty",
        torch_op=torch.empty,
        dtypes=consts.FLOAT_DTYPES,
        case_fn=empty_case_fn,
        build_inputs_fn=empty_build_inputs_fn,
    )
    bench.run()
