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

import os
import sys

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base

# Same package-resolution bootstrap as the correctness suite: the benchmark
# package that ships with this file must win over any other top-level
# ``benchmark`` package already importable on sys.path (the KernelGen harness
# runs pytest in-process with its own ``benchmark`` package).
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_BENCH_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
_IMPORTED_BENCHMARK = sys.modules.get("benchmark")
if _IMPORTED_BENCHMARK is not None and os.path.abspath(
    getattr(_IMPORTED_BENCHMARK, "__file__", "")
) != os.path.join(_BENCH_DIR, "__init__.py"):
    del sys.modules["benchmark"]


# ``_empty_affine_quantized`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# markers directly on the MarkGenerator so ``@pytest.mark._empty_affine_quantized``
# and ``-m _empty_affine_quantized`` both work.
for _name in ("_empty_affine_quantized", "_empty_affine_quantized_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_empty_affine_quantized is a factory that returns a fresh per-tensor
# affine quantized tensor with uninitialized storage, so the benchmark measures
# dispatch + storage-construction overhead rather than memory bandwidth. The
# default shape set contains a 1-B-element 1-D tensor whose cost would be
# dominated by input allocation; use allocation-friendly shapes that still
# exercise a realistic range of ranks.
EMPTY_AFFINE_QUANTIZED_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (20, 320, 15),
    (16, 128, 64, 1280),
]

# Quantized storage dtypes; the .out variant exercises the qparam-reset path in
# addition to the fill/construction work.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

SCALE = 0.1
ZERO_POINT = 0


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={"scale": SCALE, "zero_point": ZERO_POINT},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    return shape, {
        "dtype": dtype,
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
        "device": device,
    }


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=device, scale=1.0, zero_point=0
    )
    return shape, {
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
        "out": out,
    }


class EmptyAffineQuantizedBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes.

    The default shape set contains a 1-B-element 1-D tensor whose cost would be
    dominated by input allocation, so the case list is restricted to the shapes
    above.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = EMPTY_AFFINE_QUANTIZED_SHAPES


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized():
    bench = EmptyAffineQuantizedBenchmark(
        op_name="_empty_affine_quantized",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._empty_affine_quantized,
        gems_op=getattr(flag_gems, "_empty_affine_quantized", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out():
    bench = EmptyAffineQuantizedBenchmark(
        op_name="_empty_affine_quantized.out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._empty_affine_quantized.out,
        gems_op=getattr(flag_gems, "_empty_affine_quantized_out", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()
