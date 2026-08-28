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


def _has_compatible_shallow_copy_type_input_fn(shape, dtype, device):
    # Metadata-only check between two tensors of the same layout.
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    yield inp1, inp2


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape, 0),
    )


@pytest.mark.has_compatible_shallow_copy_type
def test_has_compatible_shallow_copy_type():
    bench = base.GenericBenchmark(
        op_name="has_compatible_shallow_copy_type",
        case_fn=_case_fn,
        build_inputs_fn=base.build_inputs_from_generic_input_fn(
            _has_compatible_shallow_copy_type_input_fn
        ),
        # Baseline is the native PyTorch op; the Gems path exercises the actual
        # FlagGems implementation so the benchmark measures gems vs torch.
        torch_op=torch._has_compatible_shallow_copy_type,
        gems_op=flag_gems._has_compatible_shallow_copy_type,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
