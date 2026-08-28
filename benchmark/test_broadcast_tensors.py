from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class BroadcastTensorsBenchmark(base.Benchmark):
    # broadcast_tensors accepts multiple tensors of different shapes,
    # so shapes here represent the target broadcast output shape.
    DEFAULT_SHAPE_DESC = "target shape"

    def set_shapes(self, shape_file_path=None):
        # Representative sizes covering 2D, 3D and a range of element counts
        # for broadcast_tensors performance measurement.
        self.shapes = [
            (64, 64),
            (256, 256),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]

    def get_input_iter(self, dtype) -> Generator:
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def supports_cases(self) -> bool:
        return type(self).get_input_iter is BroadcastTensorsBenchmark.get_input_iter

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, shape in enumerate(self.shapes):
            if len(shape) >= 2:
                shape_a = list(shape)
                shape_a[0] = 1
                shape_b = list(shape)
                shape_b[1] = 1
                input_shapes = (tuple(shape_a), tuple(shape_b))
            else:
                input_shapes = (shape, (1,))
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"inputs": input_shapes, "output": shape},
                    builder_args=input_shapes,
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape_a, shape_b = plan.builder_args
        inp1 = utils.generate_tensor_input(shape_a, case.dtype, self.device)
        inp2 = utils.generate_tensor_input(shape_b, case.dtype, self.device)
        return inp1, inp2


@pytest.mark.broadcast_tensors
def test_broadcast_tensors():
    bench = BroadcastTensorsBenchmark(
        op_name="broadcast_tensors",
        torch_op=torch.broadcast_tensors,
        gems_op=flag_gems.broadcast_tensors,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
