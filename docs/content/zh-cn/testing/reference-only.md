---
title: Reference-only 原始参考验证
weight: 50
---

# Reference-only：验证原始参考实现可执行性

`--reference-only` 在目标运行环境中执行原始 reference，不需要 candidate，不调用 override，不做候选正确性比较、warmup、计时或 Profile。它回答“原测试的输入生成和参考计算能否完成”，不是证明 reference 数学语义正确，也不是性能验收。调用方仍须单独运行候选 Preflight、正确性和 benchmark。

正确性 reference 与 timing baseline 是两个独立来源，分别执行、分别保存报告，不能用 benchmark 的 `torch_op` 代替 correctness pytest 的 reference。

## 命令

```bash
# 正确性参考：保留原 dtype、shape、精度转换、skip 和 --ref 选择。
pytest -q tests/test_negative.py --reference-only --output correctness-reference.json

# 正确性参考显式在 CPU 上运行；输入准备仍遵循源测试。
pytest -q tests/test_rsqrt.py --reference-only --ref cpu --output correctness-reference.json

# 原 benchmark baseline：复用 core case 构造，执行一次并同步，不计时。
pytest -q benchmark/test_negative.py --reference-only --level core --output timing-reference.json

# 对 benchmark 可沿用 --list-cases 得到的精确 case ID；二者分开执行。
pytest -q benchmark/test_negative.py --reference-only --level core \
  --case-id '<原 case_id>' --output timing-reference.json
```

正确性用例可用 pytest nodeid 精确选择，不另造 correctness case ID。`--reference-only` 与 `--override`、`--override-config`、`--preflight-only`、`--profile-only`、`--list-cases`、`--query`、benchmark `--parallel` 和 xdist 并发不兼容，均在加载候选前拒绝。正确性和 benchmark 不能在同一次 reference-only pytest 调用中混跑，避免两个报告覆盖同一个输出文件。默认输出为 `reference_result.json`；每次覆盖写入，不累计旧结果。

## 正确性 pytest 如何接入

任意 Python 测试无法被通用地截断为 reference-only：第一次 Gems 调用之后可能还有第二段 reference、梯度计算或输入生成。接口不分析变量名、不裁剪 AST、不把首段成功伪装为整例完成。测试必须显式添加 `@pytest.mark.reference_only`，用 `reference_call()` 调用原 reference，并在该模式下绕过所有候选及比较代码。

```python
from flag_gems.testing.reference import reference_call, reference_only

@pytest.mark.reference_only
def test_example(...):
    # 原始输入、to_reference、精度和设备处理保持不变。
    ref_out = reference_call(original_reference, ref_input)
    if reference_only():
        return  # 仅当本用例所有 reference 路径已经执行完成时才返回。
    # 原 candidate 调用和比较保持不变。
```

有多个 reference 的测试应逐段包围 candidate 分支，而不是提前 return；本分支的 `test_addmm` 保留两次原始 `torch.addmm` reference 调用。梯度 reference 同样需要显式记录原调用，不得以 forward 成功代替 backward。作者必须确认 reference 不依赖 candidate 的返回、mutation 或 RNG 副作用；尚未能分离的测试保持未适配状态，不能改写参考语义强行通过。

普通测试模式下 `reference_call` 只是调用原函数，返回值和异常不变。reference-only 模式下每次调用必须完成同步后才记为 `PASSED`；异常不会被吞掉。已声明支持却未记录任何成功调用的用例会失败。未声明支持的测试不执行其测试函数，报告 `UNSUPPORTED`，而不是复用普通 pytest 的 skip 作为成功。

初始接入范围：`test_negative`、`test_rsqrt`、`test_rsqrt_` 和 `test_addmm`（不是整个 addmm 文件的所有测试）。其他正确性测试需按原 reference 路径逐项适配；不能声称所有 Gems pytest 都已支持。

## Benchmark 支持边界

标准 case-based Benchmark 复用 `build_inputs()`、`unpack_to_args_kwargs()` 和原 `torch_op`。普通计时与 reference-only 共用同一个 forward/backward callable 构造；backward 会执行原 forward 和 `torch.autograd.grad`，但不调用计时器。reference-only 不进入 `use_gems`，不调用 `gems_op`，不使用候选 override。

原 `skip_native` 条件保留为 `SKIP`，不偷偷改为运行 candidate。没有 case builder，或自定义 `run/get_latency/_measure_input` 的 benchmark 明确报告 `UNSUPPORTED`；不能跳过其特殊 reference 逻辑后宣称已验证。若测试在标准 Benchmark 之外执行自定义代码，它仍是受信任 Python；该模式不是任意测试代码的沙箱。

## 结构化结果

```json
{
  "schema_version": "flaggems.reference/v1",
  "phase": "correctness",
  "status": "PASSED",
  "records": [
    {
      "nodeid": "tests/test_negative.py::test_negative[dtype-shape]",
      "operator": ["negative"],
      "status": "PASSED",
      "reference_calls": [{"status": "PASSED"}],
      "reason": null
    }
  ]
}
```

benchmark 报告的 phase 为 `timing`，case 记录包含原 `case_id`、调用次数和状态，不含 latency/speedup；pytest 阶段级错误或跳过另带 `pytest_phase`。报告区分 `PASSED`、`FAILED`、`UNSUPPORTED`、`ALL_SKIP`、`NO_CASES`；`PASSED` 必须存在真实完成的 reference 调用，源条件跳过单独保留。若原 pytest 在遍历中途 skip，整个 node 按原语义跳过，之前的调用只保留为执行证据，不据此声称该 node 完整通过。全部跳过沿用 pytest 的正常 skip 退出行为，但结构化状态为 `ALL_SKIP`，调用方不能只凭退出码 0 判定 reference 就绪。未适配或失败返回非零；中断、配置错误等原 pytest 退出码不被改写成成功。

KGS 后续可在设备 slot 内调用此模式并保存报告；本分支只实现 Gems 接口，没有修改 KGS/KG 调度、审核或 readiness。接口能力、支持范围和结构化结果都必须由调用方核对，不能把该标志当作已执行完整正确性测试的证明。

## 本轮验证

基于 2026-09-24 fetch 到的官方 `master@d17e23e48e26b6396bd3fc899c95dac74f305cbe`，在现有本地 `kernelgen-nvidia-cu128` 容器验证，先核对 `flag_gems.__file__` 来自本 feature。使用现有 Torch `2.11.0+cu130`，没有安装、升级依赖或改变设备配置。

`tests/core/test_reference_only.py`、`test_benchmark_preflight.py` 和 `test_benchmark_case_contract.py` 共 68 项通过。覆盖真实 pytest 子进程报告、配置互斥、源 skip/未适配/零调用/异常/teardown、case 精确选择、backward、正常模式回归，以及上述四个已适配 pytest 函数的 CPU reference 调用；精度分支包含 support_fp64 和 CPU reference 选择。

设备同步在 host 测试中模拟，没有运行 GPU/国产芯片 reference smoke、KGS 集成或模型 E2E；这不是全算子或跨平台验收。尚未适配的正确性 pytest 仍应返回 `UNSUPPORTED`。
