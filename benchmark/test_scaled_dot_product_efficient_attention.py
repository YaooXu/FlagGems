import pytest
import torch

import flag_gems

from . import base, consts

# Attention benchmark shapes: (batch, heads, seq_len, head_dim)
# Cover small-to-medium configurations typical in unit and integration tests
ATTENTION_BENCHMARK_SHAPES = [
    (1, 2, 8, 16),
    (2, 4, 16, 32),
    (4, 8, 32, 64),
    (8, 16, 64, 64),
]


class ScaledDotProductEfficientAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ATTENTION_BENCHMARK_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            batch, num_heads, seq_len, head_dim = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "batch": batch,
                        "num_heads": num_heads,
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                    },
                    params={"attn_bias": None, "compute_log_sumexp": False},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        batch, num_heads, seq_len, head_dim = plan.builder_args[0]
        query = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=case.dtype, device=self.device
        )
        key = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=case.dtype, device=self.device
        )
        value = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=case.dtype, device=self.device
        )
        return query, key, value, None, False


@pytest.mark.scaled_dot_product_efficient_attention
def test_scaled_dot_product_efficient_attention():
    bench = ScaledDotProductEfficientAttentionBenchmark(
        op_name="scaled_dot_product_efficient_attention",
        torch_op=flag_gems._scaled_dot_product_efficient_attention,
        gems_op=flag_gems._scaled_dot_product_efficient_attention,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
