---
title: KernelGen 测试对接
weight: 30
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# KernelGen 与 FlagGems 测试对接需求

## 我们的三个诉求

### 1. 统一指定“待测代码”的方式

本文所说的“待测代码”，是指 KernelGen Agent 当前生成、尚未合入 FlagGems 的算子实现。Agent 会连续生成和测试多个版本，因此需要在不手动替换 FlagGems 源文件的情况下，告诉测试框架“本轮测试使用哪一份待测代码”。建议把它设计成 pytest 的统一命令行参数：

```bash
python3 -m pytest tests/test_clamp_max.py -m clamp_max --candidate-code-path /path/to/main.py
```

同一个参数同时适用于正确性测试和性能测试。待测文件按约定导出 `run` 函数；测试框架在本次 pytest 进程内加载它，并把它接到所选算子的直接调用入口。pytest 退出后替换自动失效，不修改 FlagGems 源文件，也不影响下一次测试。不传 `--candidate-code-path` 时仍使用 FlagGems 默认实现，原有 CI 行为保持不变。

一次带待测代码的 pytest 命令只允许选择一个明确的算子。如果选中的测试对应多个算子、待测文件没有导出约定入口，或者测试结束时发现待测代码实际没有被调用，pytest 应直接报错，避免出现“命令成功但测到的不是待测代码”。

测试内部仍需要有统一的直接调用链：正确性测试通过 `resolve_gems_op()` 获取实现，性能测试由 Benchmark 基类根据 `op_name` 解析实现。测试文件中的 `gems_op=flag_gems.<op>` 只描述没有传入待测代码时使用的默认 FlagGems 函数，不再作为 KernelGen 指定待测代码的外部接口。

自动化系统可以额外传入 `--candidate-report-path /path/to/report.json`，获得待测文件绝对路径、SHA256、实际绑定的算子、总调用次数以及每个 pytest node 和性能 `case_id` 的调用覆盖。该报告用于正确性、preflight 和精确重放检查；启用报告会安装逐调用计数 wrapper，因此正式性能计时只传 `--candidate-code-path`，由 benchmark 结果中的 `candidate_source=override` 确认路由，避免把 Python 计数开销带入 latency。普通开发者仍只需要使用 `--candidate-code-path`。

### 2. 能够识别并精确重放单个 Workload

一个算子测试通常包含多组输入，例如不同的数据类型、形状和参数。本文将其中一组具体输入称为一个 Workload。KernelGen 需要为每个 Workload 获得稳定、唯一的标识，并能够单独重新生成和执行这一组输入，而不是每次都重新运行整个算子的全部测试。

这项能力用于复现失败、比较优化前后的结果，以及对某一个性能较差的 Workload 做进一步分析。单独重放时使用的输入生成逻辑和判定标准必须与完整测试一致。

### 3. 支持仅运行“待测代码”

Agent 需要能够跳过 PyTorch reference 和完整性能对比，只运行本轮待测代码。正式评测前，对全部 core Workload 各运行一次，用于快速确认待测代码能够编译和执行；发现性能问题后，只重复运行指定的一个 Workload，供 profiler 采集信息和定位耗时。执行结果应以机器可读的形式返回，使 Agent 能确认实际运行的是本轮待测代码，并判断哪些 Workload 成功或失败。

## 背景：我们在解决什么问题

FlagGems 维护算子的正式正确性测试和性能测试，KernelGen Agent 负责生成和优化新的算子代码，最终产物仍要提交回 FlagGems，并通过其正式评测和 CI。这里的 CI 指 FlagGems 合入代码前运行的自动化检查，其中会用 `pytest` 逐个检查算子的正确性；性能则由 FlagGems 的 benchmark 评测。

我们的核心目标是：**让优化阶段和最终提交阶段尽量使用同一套测试定义、输入和判定标准，避免待测代码在 KernelGen 中通过，提交到 FlagGems 后却得到不同结果。**

Agent 的典型工作流程如下：

```text
生成待测代码 -> 快速试跑 -> 正确性和性能测试 -> 分析结果 -> 对指定 Workload 做性能分析 -> 修改代码 -> 再次测试
```

这个流程会频繁测试不同版本的待测代码，也会多次定位并重放某一个 Workload。如果每一轮都手动替换 `ops` 目录中的实现，不仅自动化成本高，也容易出现文件未恢复、测错代码或优化评测与最终评测不一致的问题。

## 当前测试方式为什么需要统一

FlagGems 现有测试并不是全部通过同一种方式找到被测实现：一部分测试先调用 PyTorch 接口，再由框架自动路由到 FlagGems 实现；部分性能测试则通过 `gems_op` 保存默认 FlagGems 函数。直接调用函数可能绕过原有自动路由过程中的参数处理，因此两种调用方式的结果不能直接假定完全相同。

性能测试能够接收 `gems_op`，说明它已经具备明确的默认算子入口，但这还不是 Agent 所需的外部接口：KernelGen 仍需要在 pytest 启动时通过 `--candidate-code-path` 换成本轮生成的代码；正确性测试也需要接入同一解析链。现有接口还没有完整提供单个 Workload 的稳定标识、重放和结构化结果。

过去 KernelGen 若要评测新生成的实现，主要依赖手动替换 FlagGems `ops` 目录中的文件，再运行原有测试。这样可以让原有路由机制找到新代码，但不适合 Agent 高频、自动地测试多个版本，也难以只针对一个 Workload 做分析。

## 希望 FlagGems 团队配合确认的事项

1. 确认上述三项能力可以作为 FlagGems 测试框架的公共扩展，并共同确定接口边界和默认行为。
2. 确认哪些 KernelGen 相关的 `pytest` 需要逐步改为统一指定待测代码，并评审这些测试的迁移方式。
3. 使用 FlagGems 正式 CI 对迁移前后的相同用例做对比，确认原有默认测试结果没有非预期变化。

KernelGen 团队负责提供接口实现、测试迁移和沐曦机器上的回归结果；FlagGems 团队主要负责确认接口是否适合长期维护，以及通过正式 CI 判断改动能否合入。

## 是否合入的影响

### 如果不合入

**KernelGen 将长期维护一套只能在修改版 FlagGems 上运行的优化测试，优化阶段与最终提交阶段的评测可能不一致。**

KernelGen 后续编写的测试会统一依赖 `--candidate-code-path` 和直接算子解析链，但正式提交到 FlagGems 时仍要切换回主分支现有的调用方式。这意味着同一份算子代码需要经过两套路径验证，既增加适配和重复测试的工作量，也可能出现优化时通过、提交时失败，或者两边性能结论不同的情况。

此外，KernelGen 需要的 Workload 标识、单独重放和输入生成接口如果只存在于修改版 FlagGems 中，相关测试和工具便无法直接在主分支运行，后续每次同步或提交都需要额外改写。

### 如果合入

**整体风险可控，但不是零风险：部分现有测试需要接入统一的待测代码入口，调用路径变化可能改变原有测试结果。**

主要风险不在 `--candidate-code-path`、Workload 标识、重放或结果报告本身，这些能力均为显式启用，不改变原有默认测试流程。需要重点验证的是迁移到直接算子解析链的 `pytest`：原有测试可能经过框架自动路由，改成直接调用函数后，参数处理和执行路径可能存在差异。沐曦机器的抽样对比已经观察到少量原本通过的用例在改为直接调用后失败，因此这项风险需要承认并通过回归测试控制。

建议只迁移 KernelGen 实际需要的测试，每一批迁移都对比修改前后的相同用例，并以 FlagGems 正式 CI 结果作为最终合入标准。FlagGems 后续需要维护的是少量公共接口和对应回归测试，而不是一套独立的 KernelGen 测试框架。

## 建议的合入方式

1. 先加入默认关闭的 `--candidate-code-path`，并选择少量代表性算子验证“不传参数时保持原样、传入参数时只运行指定待测代码”。
2. 再补充 Workload 标识、单独重放和结构化结果，验证单独执行与完整测试使用相同输入和判定标准。
3. 最后分批迁移 KernelGen 必需的 `pytest`，每批都运行修改前后对比和 FlagGems 正式 CI，避免一次性扩大影响范围。

合入验收的核心标准是：不启用新能力时，原有测试结果保持不变；启用后，Agent 能确认实际执行了指定的待测代码，并能够稳定重放任意一个 Workload。

## 附录：后续新增正确性和性能 pytest 文件规范

### 适用范围和统一原则

**从本规范生效后，KernelGen 统一通过 pytest 的 `--candidate-code-path` 指定待测代码，所有新写的正确性测试和性能测试都必须接入同一条直接算子解析链。** 正确性测试在测试函数内通过 `resolve_gems_op()` 得到实际实现；性能测试在创建 Benchmark 时显式传入默认的 `gems_op=flag_gems.<op>`，再由 Benchmark 基类统一解析命令行传入的待测代码。这两类新文件中不得再新增依赖 `with flag_gems.use_gems()`、PyTorch dispatcher 或手动替换 `ops` 文件的测试用例；dispatcher 兼容性如需验证，应作为独立集成检查，不替代本规范中的正确性和性能测试。

这项规范适用于后续新增文件和重写文件，不要求一次性改完所有历史测试。历史测试仍按前述方式分批迁移并执行修改前后对比。

每个新增算子原则上同时提供以下两个文件：

| 文件 | 目的 | 统一的待测代码入口 |
| --- | --- | --- |
| `tests/test_<op>.py` | 验证数值、返回值、dtype、shape、原地修改等语义 | 在每个测试函数内调用 `resolve_gems_op()`；它会选择 CLI 待测代码或默认实现 |
| `benchmark/test_<op>.py` | 比较 PyTorch reference 与待测代码的性能，并提供 Workload 枚举、重放和 profiling | 创建 Benchmark 时传入默认 `gems_op`，由基类统一解析 CLI 待测代码 |

两个文件中的算子名必须一致：文件名、pytest marker、`resolve_gems_op()` 的名称、Benchmark 的 `op_name` 和 `gems_op` 应指向同一个公开算子。原地版本、out 版本和 backward 版本视为不同算子，例如 `clamp_max` 与 `clamp_max_` 分别使用各自的名称和入口。

### 正确性测试规范

正确性测试的一个 pytest 参数组合就是一个独立 Workload。shape、dtype、标量和其他算子参数应使用 `pytest.mark.parametrize` 明确展开，不要在一个测试函数内部用循环隐藏多组互不相同的 Workload。复杂参数如果无法生成稳定易读的 pytest nodeid，应显式提供稳定的 `ids`。

正确性测试必须遵守以下要求：

1. `gems_op` 必须在测试函数内解析，不得在模块导入阶段缓存。这样 `--candidate-code-path` 指定的待测代码才能在本次 pytest 中生效；不传参数时同一测试仍调用默认实现。
2. reference 使用 PyTorch 原生实现，待测结果只通过 `gems_op` 获得。不得在待测路径中调用 `torch.ops.*`、`torch.<op>` 或 `use_gems()`，也不得在异常后回退到 PyTorch 实现。
3. 输入 Tensor 在 `flag_gems.device` 上生成，并使用 `accuracy_utils.to_reference()` 构造 reference 输入。对于原地修改、out、alias 或可能修改输入的算子，reference 与待测代码必须使用内容相同但存储独立的输入，并同时检查返回值和被修改对象。
4. 传给直接函数的参数类型必须符合 FlagGems 公开 Python 接口。普通标量使用 Python 的 `int`、`float` 和 `bool`，不得依赖 dispatcher 把 NumPy 标量或其他特殊对象自动转换成 schema 类型。
5. 浮点结果使用 `gems_assert_close()`，整数、布尔值或必须逐位一致的结果使用 `gems_assert_equal()`。只有算子语义确有需要时才能调整容差，并在代码中说明原因；tuple、多输出、dtype、shape、mutation 和 alias 语义需要分别检查。
6. shape 和 dtype 优先复用 `accuracy_utils` 中的公共集合。新增本地 case 必须覆盖该算子的关键边界，而不是重复维护一套与正式测试不同的常规 shape。
7. 随机算子应固定测试所需的随机状态，并按算子语义选择确定性、边界或统计检查。不得通过放宽断言、重试直到通过或无理由 skip 来掩盖随机失败。
8. 厂商限制只能使用带明确原因的 `pytest.mark.skipif` 或 `xfail` 表达，reason 应关联已知问题或清楚说明硬件限制。不得因为当前待测代码失败而新增平台 skip。

下面以 `clamp_max` 为例展示推荐形态；其他算子应保留相同结构并替换输入、reference 和断言：

```python
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("maximum", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max(shape, maximum, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp_max(ref_inp, maximum)
    gems_op = flag_gems.testing.resolve_gems_op("clamp_max", flag_gems.clamp_max)
    res_out = gems_op(inp, maximum)

    utils.gems_assert_close(res_out, ref_out, dtype)
```

对于原地算子，reference 输入必须先复制，并检查输入修改后的值。例如：

```python
ref_inp = utils.to_reference(inp.clone())
ref_out = ref_inp.clamp_max_(maximum)
gems_op = flag_gems.testing.resolve_gems_op("clamp_max_", flag_gems.clamp_max_)
res_out = gems_op(inp, maximum)

utils.gems_assert_close(res_out, ref_out, dtype)
utils.gems_assert_close(inp, ref_inp, dtype)
```

### 性能测试规范

性能测试必须使用 FlagGems 的 Benchmark 基类完成 reference、warmup、计时和结果记录，不得在 pytest 文件中自行编写另一套计时循环。`torch_op` 表示性能对照实现，测试文件传入的 `gems_op` 表示 FlagGems 默认实现；传入 `--candidate-code-path` 后，基类应统一把实际被测实现换成待测文件中的 `run`。对照实现、默认实现和待测实现必须具有相同的输入及调用语义。

新增性能测试必须遵守以下要求：

1. Benchmark 形式按三层选择：优先使用已经覆盖算子语义的公共 Benchmark family，例如 `UnaryPointwiseBenchmark`、`BinaryPointwiseBenchmark`、`UnaryReductionBenchmark` 或 `BlasBenchmark`；公共 family 无法表达输入时使用两阶段 `GenericBenchmark`；输入结构、shape 规则或指标计算仍无法表达时，才自定义 Benchmark 子类。
2. 创建 Benchmark 时必须显式设置 `op_name`、`torch_op`、`gems_op` 和 `dtypes`。原地、out、backward 等特殊语义还必须设置对应的 Benchmark 参数，例如 `is_inplace=True`。
3. 所选 Benchmark 必须支持 Workload 枚举和精确重放。公共 family 已经提供该能力时直接复用；使用 `GenericBenchmark` 自定义输入时，必须同时提供 `case_fn` 和 `build_inputs_fn`，不得只提供旧式 `input_fn`。
4. `case_fn` 只描述 Workload，不得创建 Tensor、访问设备或执行算子。它必须按确定顺序生成 `BenchmarkCasePlan`，并在 `shape` 和 `params` 中记录能够让人理解该 Workload 的 JSON 可序列化信息。
5. `build_inputs_fn` 只为选中的一个 Workload 创建真实输入，并返回 Benchmark 调用所需的 positional arguments 和 keyword arguments。完整 benchmark、Preflight、单 Workload 重放和 profiling 必须共用这一份输入构造逻辑。
6. Workload 顺序必须稳定。不得使用未固定种子的全局随机数决定 case 内容；如果参数需要伪随机生成，应使用由稳定 case 信息构造的局部随机种子。测试合入后不要随意重命名测试函数或重排已有 case，新增 case 原则上追加在原序列末尾，避免已有 `case_id` 改变含义。
7. 性能输入应覆盖正式关心的 core shape、dtype、布局和关键参数，但不要把正确性测试的全部组合机械复制到 benchmark。性能测试只保留能够代表实际负载或关键性能分支的 Workload。

使用公共 Benchmark family 时，推荐形态如下：

```python
import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.acosh
def test_acosh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh",
        torch_op=torch.acosh,
        gems_op=flag_gems.acosh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

输入或参数无法由公共 family 表达时，使用两阶段的 `GenericBenchmark`：

```python
import pytest
import torch

import flag_gems

from . import base, consts, utils


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"max": 3.14},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"max": plan.params["max"]}


@pytest.mark.clamp_max
def test_clamp_max():
    bench = base.GenericBenchmark(
        op_name="clamp_max",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.clamp_max,
        gems_op=flag_gems.clamp_max,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

`BlasBenchmark` 等少数公共 family 本身要求传入专用 `input_fn`，只要该 family 已经统一实现 Workload 枚举和重放，就不属于“仅使用旧式 `input_fn`”的禁止情形。

### 自定义 Benchmark 规范

现有性能测试中确实存在直接继承 `base.Benchmark` 或公共 family 的 custom Benchmark，例如算子需要特殊 shape、多个关联 Tensor、非标准输出，或者需要自定义 GB/s、TFLOPS 等指标。后续仍允许这种形式，但 custom Benchmark 只负责表达该算子的特殊输入和指标，不能重新实现一套候选代码选择、warmup、计时、结果记录或 profiling 流程。

自定义 Benchmark 必须满足以下要求：

1. 仍由 `base.Benchmark.run()` 驱动，pytest 中仍显式传入默认 `gems_op`，实际待测代码由基类根据 `--candidate-code-path` 统一解析。除非公共框架本身缺少必要能力，否则不得覆盖 `run()`、待测代码解析或计时主流程。
2. 必须同时实现 `get_case_iter(dtype)` 和 `build_inputs(case)`：前者只生成不含 Tensor 的稳定 Workload 描述，后者只构造选中的一组真实输入。不得只实现旧式 `get_input_iter()`。
3. 需要特殊 shape 时可以覆盖 `set_shapes()` 或 `set_more_shapes()`；需要特殊指标时只覆盖 `get_gbps()`、`get_tflops()` 等指标钩子。shape、参数和指标定义必须与算子实际语义一致。
4. 继承某个公共 family 时，如果覆盖了该 family 原有的 case 循环，也必须同步覆盖 Workload 枚举和输入构造，不能让完整 benchmark 与单 case 重放走不同路径。

推荐骨架如下：

```python
import pytest
import torch

import flag_gems

from . import base


AFFINE_GRID_SHAPES = [(1, 4, 4), (2, 8, 8), (4, 16, 16)]


class AffineGridBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = AFFINE_GRID_SHAPES

    def get_case_iter(self, dtype):
        for ordinal, (n, h, w) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"theta": (n, 2, 3)},
                    params={"size": [n, 3, h, w], "align_corners": False},
                    builder_args=(n, h, w),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        n, h, w = plan.builder_args
        theta = torch.randn((n, 2, 3), dtype=case.dtype, device=self.device)
        return theta, plan.params["size"], plan.params["align_corners"]


@pytest.mark.affine_grid_generator
def test_affine_grid_generator():
    bench = AffineGridBenchmark(
        op_name="affine_grid_generator",
        torch_op=torch.affine_grid_generator,
        gems_op=flag_gems.affine_grid_generator,
        dtypes=[torch.float32],
    )
    bench.run()
```

### 新增 pytest 的验收标准

每组正确性和性能测试至少完成以下验证后才能提交评审：

1. 不传 `--candidate-code-path` 时，正确性和性能测试调用 FlagGems 默认实现并全部通过；传入参数后，所有相同 Workload 都改为执行待测文件中的 `run`，pytest 结束后再次运行能够恢复默认实现。
2. 性能测试能够在不创建 Tensor 的情况下列出全部 Workload，每个 `case_id` 唯一且信息可读；任意选择一个 `case_id` 都能单独重放，并与完整 benchmark 中的对应 Workload 使用相同输入构造逻辑。
3. Preflight 能让每个 core Workload 各执行一次待测 `gems_op`；profiling 只执行指定的一个 Workload；正式性能结果能确认本轮使用的待测代码。
4. 正确性与性能文件中的算子名称、参数语义和 dtype 范围一致。正确性覆盖可以更广，但不能出现同名算子在两个文件中采用不同参数含义的情况。
5. 新增测试通过目标设备回归。若是从 dispatcher 或旧式 `input_fn` 迁移的历史测试，还必须保存修改前后的同 case 对比，并以 FlagGems 正式 CI 结果作为最终判断。

建议在目标设备上至少执行以下检查。下面以 `clamp_max` 为例，最后两条命令中的 `case_id` 来自第一条性能命令生成的 JSON：

```bash
python3 -m pytest -q tests/test_clamp_max.py --quick
python3 -m pytest -q tests/test_clamp_max.py --quick --candidate-code-path /path/to/main.py
python3 -m pytest -q tests/test_clamp_max.py --candidate-code-path /path/to/main.py
python3 -m pytest -q benchmark/test_clamp_max.py --level core --list-cases --output /tmp/clamp-max-cases.json
python3 -m pytest -q benchmark/test_clamp_max.py --level core --preflight-only --candidate-code-path /path/to/main.py
python3 -m pytest -q benchmark/test_clamp_max.py --level core --case-id 'benchmark/test_clamp_max.py::test_clamp_max::core::float16::0' --record json --output /tmp/clamp-max-result.json --candidate-code-path /path/to/main.py
python3 -m pytest -q benchmark/test_clamp_max.py --level core --profile-only --case-id 'benchmark/test_clamp_max.py::test_clamp_max::core::float16::0' --candidate-code-path /path/to/main.py
```

代码评审时，如果新正确性测试仍包含依赖 `use_gems()` 或 dispatcher 的算子测试、性能 Benchmark 没有显式传入默认 `gems_op`、测试不能通过 `--candidate-code-path` 执行待测代码、`GenericBenchmark` 没有使用两阶段输入、自定义 Benchmark 只实现 `get_input_iter()` 而无法枚举和重放 Workload，或者测试通过 fallback、无理由 skip 和放宽断言掩盖失败，应直接退回修改。
