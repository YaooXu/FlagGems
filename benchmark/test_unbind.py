import pytest
import torch

import flag_gems

from . import base, consts

UNBIND_SHAPES = [
    (2, 3),
    (4, 8),
    (16, 32),
    (4, 8, 16),
    (32, 64, 128),
    (2, 4, 8, 16),
]


class UnbindBenchmark(base.Benchmark):
    """Benchmark for unbind operation (zero-copy view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = UNBIND_SHAPES

    def get_input_iter(self, dtype):
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dim = 0
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"dim": dim},
                    builder_args=(shape, dim),
                ),
            )

    def build_inputs(self, case):
        shape, dim = case.builder_args[0].builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, dim


@pytest.mark.unbind
def test_unbind():
    bench = UnbindBenchmark(
        op_name="unbind",
        torch_op=torch.unbind,
        gems_op=flag_gems.unbind,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
