---
title: Benchmark reference-only
weight: 50
---

# Benchmark reference-only：检查原始 core baseline

`--reference-only` 仅用于 benchmark，在目标运行环境中执行原始 baseline，不需要 candidate，不使用 override，不做 warmup、计时、正确性比较或 Profile。默认 review 使用 `--level core`，检查原测试输入生成和性能 reference 能否运行，不能据此声称正确性 pytest 的 reference 或完整测试链路已经通过。

```bash
pytest -q benchmark/test_negative.py --reference-only --level core --output benchmark-reference.json
```

可沿用单独一次 `--list-cases` 得到的 `--case-id` 精确重放；KGS review 会使用整个 core 集合，不缩减 dtype 或 workload。输出默认是 `reference_result.json`，每次覆盖写入，不累计旧结果。先核对 `flag_gems.__file__` 来自预期 checkout，避免旧 editable install 导致使用另一版本。

## 复用与边界

标准 case-based Benchmark 复用 `build_inputs()`、`unpack_to_args_kwargs()` 和原 `torch_op`。普通计时与 reference-only 共用 forward/backward callable 构造；backward 执行原 forward 和 `torch.autograd.grad`，但不调用计时器。每个 case 执行一次并同步，结束后释放输入和计算图，不进入 candidate 的 `use_gems`、`gems_op` 或 override 分支。

没有 case builder，或自定义 `run/get_latency/_measure_input` 的 benchmark 明确报告 `UNSUPPORTED`，不能绕过特殊 baseline 逻辑后宣称已验证。原 `skip_native` 和 pytest skip 条件保留。该模式不是任意 Python 测试代码的沙箱。

本接口不修改或执行正确性 pytest，不提供 correctness reference 的 marker、包装器或截断逻辑。正确性测试继续做源码 review，生成候选后运行原完整正确性测试。性能和正确性 reference 的 dtype、精度、设备、shape 可能不同，不能相互替代。

`--reference-only` 与 `--override`、`--override-config`、`--preflight-only`、`--profile-only`、`--list-cases`、`--query`、benchmark `--parallel` 及 xdist 并发互斥。不能与 `tests/` 下的正确性用例混跑。

## 报告

```json
{
  "schema_version": "flaggems.reference/v1",
  "phase": "timing",
  "status": "PASSED",
  "records": [{
    "nodeid": "benchmark/test_negative.py::test_negative",
    "operator": "negative",
    "case_id": "benchmark/test_negative.py::test_negative::core::float32::0",
    "count": 1,
    "status": "PASSED"
  }]
}
```

报告区分 `PASSED`、`FAILED`、`UNSUPPORTED`、`ALL_SKIP`、`NO_CASES`，不含 latency/speedup。pytest 阶段级失败或跳过另带 `pytest_phase`。原 pytest 中途 skip 时，整个 node 按源语义跳过，之前的调用只保留为执行证据，不据此声称该 node 完整通过。全部跳过可能仍为 pytest exit code 0，因此调用方必须检查结构化状态，不能仅凭进程退出码判断就绪。

KGS 配套通过设备 slot 和隔离 worker 调用此命令，核对冻结 benchmark fingerprint 与 core case 覆盖，保存原报告；该执行事实不是模型审核结论，也不写入候选优化 ledger。KG 的 `skip_review` 仅跳过模型审核，不跳过这项目标验证。

## 验证范围

分支基于官方 `master@d17e23e48e26b6396bd3fc899c95dac74f305cbe`，已撤回早期 correctness reference-only 试验，对应正确性 pytest 与该 master 保持一致。Host 测试使用现有 `kernelgen-nvidia-cu128` 容器，验证原 baseline、backward、精确 case 选择、skip/异常、配置互斥与既有 Preflight/Profile 路径；设备同步使用模拟实现。未安装或升级依赖，未进行 GPU/跨芯片验收。
