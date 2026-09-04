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

## 本次公共能力 PR 的范围

本次分支只提交两层公共能力及其框架级回归测试：第一层是待测代码的进程内加载、统一解析和调用证明，第二层是性能 Workload 的无 Tensor 枚举、精确重放、Preflight 和 Profile replay。具体算子的正确性与性能 pytest 改写不在本次分支中，将在公共接口确认并合入后作为后续 PR 分批提交。

本文中的 `clamp_max`、`acosh` 和 `affine_grid_generator` 代码与命令用于说明具体算子完成后续迁移时如何接入，并不表示本次分支已经修改这些算子的 pytest。当前分支可直接评审和验证的代码边界如下：

| 提交层次 | 主要文件 | 对默认测试的影响 |
| --- | --- | --- |
| 待测代码加载与解析 | `src/flag_gems/testing/candidate.py`、`src/flag_gems/testing/pytest_plugin.py`、`src/flag_gems/testing/__init__.py`、两个 pytest `conftest.py` | 不传 `--candidate-code-path` 时不创建 candidate session；已有未接入 resolver 的算子测试继续走原路径 |
| Benchmark Workload 协议 | `benchmark/base.py`、`benchmark/consts.py`、`benchmark/summary_for_plot.py` | 已支持的公共 Benchmark family 改为等价的两阶段输入流程；尚未迁移的自定义循环继续全量运行，但显式使用 case 相关参数时会拒绝执行 |

对应的公共回归测试是 `tests/test_candidate_code.py` 和 `benchmark/test_benchmark_case_api.py`。本次分支在 NVIDIA 开发容器中运行结果为 `27 passed`，并额外验证了真实 `benchmark/test_mm.py::test_mm --list-cases` 输出、未知 `case_id` 拒绝以及 master 原有 `--mm-layout` 行为。

## 我们的三个诉求

### 1. 统一指定“待测代码”的方式

本文所说的“待测代码”，是指 KernelGen Agent 当前生成、尚未合入 FlagGems 的算子实现。Agent 会连续生成和测试多个版本，因此需要在不手动替换 FlagGems 源文件的情况下，告诉测试框架“本轮测试使用哪一份待测代码”。当前公共接口将它设计为 pytest 的统一命令行参数；以下命令要求目标正确性测试已经完成后续 resolver 迁移：

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

## 三项诉求的当前代码实现

当前分支已经把三项能力拆成可独立评审的公共组件。待测代码加载与解析位于 `src/flag_gems/testing/`，正确性测试只需接入统一 resolver；Workload 枚举、重放和 candidate-only 执行位于 `benchmark/`。因此 KernelGen 不需要修改 FlagGems dispatcher，也不需要在每轮测试前后替换 `ops` 文件。

| 职责 | 实现位置 | 作用 |
| --- | --- | --- |
| pytest 参数和 Session 生命周期 | `src/flag_gems/testing/pytest_plugin.py` | 注册 `--candidate-code-path` 等参数，在 pytest 启动时加载待测文件，在结束时校验调用覆盖并清理 |
| 待测文件加载和调用计数 | `src/flag_gems/testing/candidate.py` | 加载 `run()`、计算源码 SHA256、限制一次 Session 只绑定一个算子、记录 pytest node 和 `case_id` |
| 统一算子解析 | `src/flag_gems/testing/__init__.py` | 提供 `resolve_gems_op()`、`override_gems_op()`、`gems_op_case()`，让正确性和性能测试共享同一入口 |
| pytest 插件接入 | `tests/conftest.py`、`benchmark/conftest.py` | 让正确性和性能 pytest 都能识别同一组待测代码参数 |
| Workload 数据结构 | `benchmark/consts.py` | 定义不含 Tensor 的 `BenchmarkCasePlan`、可序列化的 `BenchmarkCaseSpec` 和结果中的 `case_id` |
| Workload 执行状态机 | `benchmark/base.py` | 实现列举、筛选、构造输入、正常计时、Preflight 和 Profile replay |
| 公共能力回归测试 | `tests/test_candidate_code.py`、`benchmark/test_benchmark_case_api.py` | 验证加载/恢复、单算子限制、调用覆盖、无 Tensor 列举和精确重放 |

当前实现状态如下：

| 诉求 | 当前状态 | 边界 |
| --- | --- | --- |
| 统一指定待测代码 | 已实现 | 一次 pytest Session 只允许一份 `run()` 绑定一个公开算子 |
| 稳定识别并重放 Workload | 已实现 | 正确性测试使用 pytest nodeid；性能测试使用 `case_id`。旧式、尚未迁移的 `get_input_iter()` Benchmark 不支持 `--list-cases`、`--case-id`、`--preflight-only` 和 `--profile-only` |
| 仅运行待测代码 | 已实现 candidate-only 的 Preflight 和单 Workload Profile replay | 已有调用覆盖 JSON；当前没有独立的“每个 case 成功/失败明细”Schema，失败仍主要通过 pytest exit status 和错误信息返回 |

### 诉求一：`--candidate-code-path` 如何替换实际被测函数

#### 1. pytest 启动时安装进程内候选 Session

`tests/conftest.py` 和 `benchmark/conftest.py` 都声明同一个内置插件：

```python
pytest_plugins = ("flag_gems.testing.pytest_plugin",)
```

插件在 `pytest_addoption()` 中注册统一参数，并在 `pytest_configure()` 中进入 candidate context。以下代码片段只保留与调用链有关的关键分支，错误包装和清理保护以源码为准：

```python
group.addoption(
    "--candidate-code-path",
    dest="candidate_code_path",
    default=None,
)

def pytest_configure(config):
    path = config.getoption("candidate_code_path")
    if not path:
        return
    context = flag_gems.testing.candidate_code(path)
    context.__enter__()
    config._flag_gems_candidate_context = context
```

不传参数时，`pytest_configure()` 直接返回，不创建 candidate session。公共框架的默认解析行为因此保持不变。

#### 2. 加载文件并校验统一入口

待测文件必须是一个真实 Python 文件，并导出 callable `run`：

```python
# /path/to/main.py
def run(inp, maximum):
    # KernelGen 生成的实现
    ...
```

`candidate.py::_load_candidate()` 读取源文件、计算 SHA256，并使用只和该文件内容关联的临时模块名加载代码：

```python
source = path.read_bytes()
digest = hashlib.sha256(source).hexdigest()
module_name = f"_flag_gems_candidate_{digest[:16]}"
spec = importlib.util.spec_from_file_location(module_name, path)
spec.loader.exec_module(module)
function = getattr(module, "run", None)
if not callable(function):
    raise RuntimeError("candidate code must export a callable run()")
```

加载期间会把待测文件所在目录临时加入 `sys.path`，因此 `main.py` 可以导入同目录的辅助模块。pytest 结束后，插件在 `pytest_unconfigure()` 中退出 context，同时移除活动 Session、临时模块和该 `sys.path` 项；替换只影响当前 pytest 进程，不写 FlagGems 源文件。

#### 3. 第一次 resolver 调用确定本次测试的算子

当前 CLI 没有额外要求用户传 `--candidate-op`。待测代码绑定的算子名由本次 pytest 中第一次 `resolve_gems_op(operator, default)` 调用确定，后续只能继续解析同一个名字：

```python
def bind(self, operator):
    if self.operator is None:
        self.operator = operator
    elif self.operator != operator:
        raise RuntimeError("--candidate-code-path may target only one public operator")
    return self.wrapped if self.track_calls else self.function
```

这意味着 `pytest -m clamp_max` 只负责筛选测试，真正绑定的 canonical operator 是测试代码传给 `resolve_gems_op()` 的 `"clamp_max"`。如果 marker 误选了另一个也会解析 `clamp_max_` 的测试，第二次解析会直接失败，不会让同一份 `run()` 静默替换两个不同签名的算子。

#### 4. 正确性和性能测试在 resolver 处汇合

`resolve_gems_op()` 的决策逻辑是：有 CLI candidate 时返回该文件的 `run()`；否则使用显式的进程内 override；两者都没有时返回测试传入的 FlagGems 默认函数。CLI candidate 与 `override_gems_op()` 同时作用于同一个算子会直接报冲突。

```python
override = _GEMS_OP_OVERRIDES.get(operator, _MISSING)
candidate = _resolve_candidate(operator)
if candidate is not None:
    if override is not _MISSING:
        raise RuntimeError("candidate and explicit override cannot be combined")
    return candidate
if override is not _MISSING:
    return override
if default is None:
    import flag_gems as package
    default = getattr(package, operator, None)
if not callable(default):
    raise LookupError(f"No direct FlagGems callable for '{operator}'.")
return default
```

这里替换的是测试的直接 callable，不修改 PyTorch dispatcher 注册表。`override_gems_op()` 是供单元测试或进程内调用者临时替换函数的 Python API；`--candidate-code-path` 由 pytest 插件管理独立的 candidate session，两者最终都通过 `resolve_gems_op()` 汇合，但 CLI 插件并不是通过修改 dispatcher 或复制 `ops` 文件完成替换。

正确性测试的调用链如下：

```python
ref_out = torch.clamp_max(ref_inp, maximum)
gems_op = flag_gems.testing.resolve_gems_op(
    "clamp_max", flag_gems.clamp_max
)
res_out = gems_op(inp, maximum)
```

性能测试先保存默认函数，Benchmark 基类在执行时再次走同一个 resolver：

```python
bench = base.GenericBenchmark(
    op_name="clamp_max",
    torch_op=torch.clamp_max,
    gems_op=flag_gems.clamp_max,
    ...,
)

# benchmark/base.py
op = flag_gems.testing.resolve_gems_op(self.op_name, self.gems_op)
source = flag_gems.testing.gems_op_source(self.op_name, op)
```

因此两条完整调用链分别是：

```text
正确性：pytest -> candidate plugin -> candidate session -> test function -> resolve_gems_op("clamp_max") -> main.py::run
性能：pytest -> candidate plugin -> candidate session -> Benchmark.run() -> _resolve_direct_gems_op() -> resolve_gems_op("clamp_max") -> main.py::run
```

#### 5. 如何证明没有“传了路径但仍测到默认实现”

正确性测试默认开启逐调用 wrapper。wrapper 从 `PYTEST_CURRENT_TEST` 取得 pytest nodeid，从 `gems_op_case()` 的 ContextVar 取得性能 `case_id`，然后累计调用次数：

```python
def tracked(*args, **kwargs):
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    nodeid = current_test.rsplit(" (", 1)[0] if current_test else None
    case_id = self.current_case(self.operator)
    self.calls[(nodeid, case_id)] += 1
    return self.function(*args, **kwargs)
```

指定 `--candidate-report-path` 后会输出类似下面的机器可读报告：

```json
{
  "schema_version": "flaggems.candidate-code-report/v1",
  "source_path": "/path/to/main.py",
  "source_sha256": "...",
  "entrypoint": "run",
  "operator": "clamp_max",
  "call_tracking": true,
  "total_calls": 12,
  "missing_nodeids": [],
  "records": [
    {
      "nodeid": "tests/test_clamp_max.py::test_clamp_max[shape0-3.14-float16]",
      "case_id": null,
      "count": 1
    }
  ]
}
```

pytest Session 结束时会检查是否从未解析公开算子、待测 `run()` 是否一次都没有调用，以及是否有未调用 candidate 的非 skip 测试节点；这些情况都会把原本成功的 pytest 改为失败。纯性能计时且未要求报告时会关闭逐调用 wrapper，避免 Python 计数进入 latency，此时结果中的 `candidate_source="override"` 表示实际函数来自进程内 candidate/override，而不是默认 `gems_op`。这里的 `override` 是结果字段的历史命名，不表示修改了 dispatcher。

`--candidate-report-path` 和 `--candidate-call-count-path` 都会显式开启逐调用 wrapper。它们适合正确性和 Preflight 验真，但不应加入正式 latency 或高保真 profiler 命令，否则 wrapper 的 Python 调用与计数也会进入被观测路径。推荐先用带报告的独立 Preflight 证明路由，再用只带 `--candidate-code-path` 的命令正式计时或 profiling；KernelGen 在外部记录本轮源码路径和 SHA256。

### 诉求二：Workload 如何枚举并精确重放

#### 正确性测试使用 pytest 原生 nodeid

正确性测试的一组参数组合天然对应一个 pytest item，因此使用 pytest nodeid 作为 Workload 标识，并直接按完整 nodeid 重放：

```python
@pytest.mark.parametrize("shape", POINTWISE_SHAPES, ids=[...])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_clamp_max(shape, dtype):
    ...
```

```bash
python3 -m pytest --collect-only -q tests/test_clamp_max.py -m clamp_max
python3 -m pytest -q '<从上一条命令复制的完整 pytest nodeid>' --candidate-code-path /path/to/main.py
```

由于 pytest nodeid 包含文件、函数和参数 ID，新增测试时必须为复杂参数提供稳定 `ids`，并避免随意重命名测试函数或改变已有参数 ID。正确性 Workload 不使用 benchmark 的 `--case-id`。

#### 性能测试使用两阶段 Case API

原有 Benchmark 常在 `get_input_iter()` 中一边循环、一边创建 Tensor。这样在不知道前面输入数量的情况下无法定位某一个内部 case，并且仅列出 case 也会占用设备。当前实现将它拆为两个阶段：

```text
case_fn()/get_case_iter()：只生成轻量描述，不创建 Tensor
                    ↓
BenchmarkCasePlan -> BenchmarkCaseSpec + 稳定 case_id
                    ↓
build_inputs()：只为选中的 case 创建真实输入
                    ↓
正常计时 / Preflight / Profile replay 共用同一份输入构造
```

核心数据结构如下：

```python
@dataclass(frozen=True)
class BenchmarkCasePlan:
    shape: dict
    params: dict = field(default_factory=dict)
    builder_args: tuple = field(default_factory=tuple, repr=False)

@dataclass(frozen=True)
class BenchmarkCaseSpec:
    case_id: str
    ordinal: int
    dtype: torch.dtype
    shape: dict
    params: dict = field(default_factory=dict)
    builder_args: tuple = field(default_factory=tuple, repr=False)

@dataclass(frozen=True)
class BenchmarkCaseList:
    op_name: str
    level: str
    cases: tuple[BenchmarkCaseSpec, ...]
    phase: str = "timing"
    schema_version: str = "flaggems.benchmark-case-list/v2"
```

`shape` 和 `params` 会写入 JSON，供人和 Agent 理解 Workload；`builder_args` 只在进程内交给 `build_inputs()`，不会进入对外 JSON。基类按下面的格式构造 ID：

```python
local_id = f"{bench_level}::{dtype_name}::{ordinal}"
case_id = f"{pytest_nodeid}::{local_id}"
```

例如：

```text
benchmark/test_clamp_max.py::test_clamp_max::core::float16::0
```

这里的稳定性来自 pytest nodeid、level、dtype 顺序和 `get_case_iter()` 的枚举顺序，不是内容哈希。因此迁移后不应重排已有 dtype、shape 或参数组合；新增 case 原则上追加到末尾。

`--list-cases` 的逻辑只调用 `get_case_iter()`，不会调用 `build_inputs()`：

```python
if Config.list_cases:
    case_list = self.list_cases(initialize=False)
    update_case_list(case_list.to_dict())
    return case_list
```

输出使用 `flaggems.benchmark-case-list/v2`，包含 `op_name`、`level` 和每个 case 的 `case_id`、`ordinal`、`dtype`、`shape`、`params`。得到 ID 后，正常性能重放通过 `--case-id` 精确筛选；`_run_cases()` 只对命中的 `BenchmarkCaseSpec` 调用 `build_inputs()` 和 `_measure_input()`。Session 结束时还会比较 requested、available 和 executed 三组 ID，未知 ID 或已选择但未执行的 ID 都会让 pytest 失败。

### 诉求三：Preflight 和 Profile 如何只运行待测代码

`benchmark/conftest.py` 注册以下互斥模式：

```text
--list-cases       只列举 Workload，不创建 Tensor、不执行算子
--preflight-only   对全部已配置 Workload 各执行一次 gems_op
--profile-only     只重放一个 --case-id，可配置 warmup 和采集次数
```

参数门禁会拒绝同时启用多个模式；`--preflight-only` 不允许再传 `--case-id`，`--profile-only` 则必须且只能传一个 `--case-id`。

`Benchmark.run()` 的分支逻辑如下：

```python
if Config.preflight_only:
    return self._run_candidate_cases(None)

if Config.profile_only:
    return self._run_candidate_cases(
        selected_case_ids,
        warmup=Config.profile_warmup,
        iterations=Config.profile_iterations,
        profile=True,
    )

return self._run_cases(selected_case_ids)
```

`_run_candidate_cases()` 仍然通过 `build_inputs()` 构造和正式 benchmark 完全相同的输入，但不会调用 `torch_op`、`get_latency()` 或 speedup 计算。Preflight 对每个 case 调用一次实际 `gems_op`；编译错误、运行时错误和输入构造错误会带 `case_id`、shape、dtype、params 和阶段信息使 pytest 失败。

Profile replay 会为 warmup 和 capture 分别构造输入，避免原地算子的 warmup 修改采集阶段输入：

```python
warmup_input = self.build_inputs(case)
capture_input = self.build_inputs(case)

for _ in range(profile_warmup):
    warmup_fn()
synchronize()

profiler_start()
for _ in range(profile_iterations):
    capture_fn()
synchronize()
profiler_stop()
```

CUDA 环境中的 `profiler_start()/stop()` 当前调用 `cudaProfilerStart()` 和 `cudaProfilerStop()`，让外部 profiler 只采集目标区间；其他芯片仍由外部厂商 profiler 围绕这条确定性 replay 命令采集，FlagGems 本身负责的是“只构造并执行指定 Workload”，不在这里实现所有厂商 profiler 的启动和结果解析。

KernelGen 的标准调用会同时传 `--candidate-code-path`；代码本身也允许不传，此时 candidate-only 状态机运行的是测试配置中的默认 FlagGems `gems_op`。因此 `--preflight-only` 和 `--profile-only` 更准确地说是“只运行被解析出的 gems_op、不运行 reference 和对比计时”，是否为外部待测代码由 `--candidate-code-path` 决定。

三种执行模式与输出的关系如下：

| 模式 | 创建 Tensor | PyTorch reference | gems_op | 计时 | 机器可读证据 |
| --- | --- | --- | --- | --- | --- |
| `--list-cases` | 否 | 否 | 否 | 否 | case list JSON |
| 正常 benchmark | 是 | 是 | 是 | 是 | benchmark JSON 中的 `case_id`、latency、speedup、`candidate_source` |
| `--preflight-only` | 是，每个 case 一次 | 否 | 是，每个 case 一次 | 否 | pytest 状态；传 `--candidate-report-path` 时包含逐 case 调用记录 |
| `--profile-only --case-id ...` | 是，只构造目标 case | 否 | 是，warmup 后执行指定次数 | 不做 benchmark 计时 | pytest 状态；传 `--candidate-report-path` 时包含目标 `case_id` 的调用记录 |

当前 candidate report 能证明哪些 case 调用了待测代码，但没有为 Preflight/Profile 定义独立的逐 case `passed/failed/error_stage` 结果 Schema；发生错误时 pytest 会在第一个失败处结束。如果双方认为批量系统必须在单次进程中收集所有 case 的完整状态，这仍是需要后续补充的接口，而不应把现有调用计数报告描述成完整测试结果。

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

性能测试能够接收 `gems_op`，说明它已经具备明确的默认算子入口，但这还不是 Agent 所需的外部接口：KernelGen 仍需要在 pytest 启动时通过 `--candidate-code-path` 换成本轮生成的代码；正确性测试也需要接入同一解析链。FlagGems 主分支原有接口没有完整提供单个 Workload 的稳定标识、重放和结构化结果，当前对接分支在此基础上增加了前述 candidate session 和两阶段 Case API。

过去 KernelGen 若要评测新生成的实现，主要依赖手动替换 FlagGems `ops` 目录中的文件，再运行原有测试。这样可以让原有路由机制找到新代码，但不适合 Agent 高频、自动地测试多个版本，也难以只针对一个 Workload 做分析。

## 当前 `gems_op` 方法无法覆盖的 pytest 类型

当前方法的基本前提是：一次 pytest 命令只选择一个 canonical operator，`--candidate-code-path` 只提供一个导出 `run` 的待测文件，这个 operator 对应一个公开的 FlagGems callable 和一套固定调用签名。此前批量转换中发现，以下三类 pytest 不满足这个前提，不能仅靠补充 `resolve_gems_op()` 或 `gems_op=...` 准确接入：

| 类型 | 已发现的例子 | 无法覆盖的原因 | 建议处理方式 |
| --- | --- | --- | --- |
| 同一个 marker 或 `op_name` 混合多个不兼容的 callable | `binary_cross_entropy`、`lu_unpack` 同时覆盖 default 与 out；`float_power_` 同时覆盖 Tensor 与 Scalar overload | 一份待测代码只有一个 `run` 入口和一套签名，不能同时替换两个参数、输出或 mutation 契约不同的公开函数 | 将 variant/overload 拆成独立 marker、`op_name` 和 resolver key；如果确实要求一次注入多个实现，则需要另行设计多入口待测代码协议 |
| pytest 覆盖的 variant 没有公开 FlagGems callable | `heaviside.out`、`_upsample_nearest_exact2d.out` 只有内部实现或未从 `flag_gems` 公共命名空间导出 | `gems_op` 必须指向可评审、可稳定调用的公共入口；不能用 default 函数冒充 out variant，也不能依赖内部模块路径，否则无法保证 caller-owned out buffer 和返回别名语义 | 先导出独立的公共 callable，再为该 variant 分配独立 marker、`op_name` 和 resolver key |
| pytest marker 不能唯一隔离目标算子 | `benchmark/test_trunc_.py` 中 `trunc` 与 `trunc_` 共用 `trunc_` marker | `pytest -m <marker>` 会同时收集目标和 sibling；同一份待测代码无法替换 sibling，调用覆盖报告也无法证明所选结果都来自目标实现 | 修正 marker 使其与 canonical operator 一一对应，或让测试协议支持按精确 pytest node/Benchmark `op_name` 选择 |

这里的限制针对的是“一个待测入口能否无歧义地替换一个被测算子”，不表示所有特殊 pytest 都不能使用当前方法。原地算子、独立的 out/backward 算子、自定义 Benchmark、旧式 `input_fn`、lambda/`functools.partial` 和 dispatcher 参数适配，只要能够保留全部 workload 和语义，并改造成一个公开 callable、一套明确签名以及统一的 Workload 枚举与输入构造，就仍然可以通过当前 `gems_op` 方法覆盖。转换时不得通过删除 workload、把 out 参数丢给 default 函数、调用未导出的内部实现或让候选代码隐藏在 wrapper 后面来规避上述限制。

## 希望 FlagGems 团队配合确认的事项

1. 确认 `pytest_plugin.py`、candidate session 和 `resolve_gems_op()` 可以作为 FlagGems 长期维护的公共测试能力，并确认“第一次 resolver 调用懒绑定算子”是否足够清晰，还是需要增加显式 operator 参数。
2. 确认“一份待测文件导出一个 `run()`，一次 pytest Session 只绑定一个 canonical operator”的协议；default、in-place、out、backward 和不兼容 overload 应使用独立公开 callable、marker、`op_name` 和 resolver key，不能由 default 加 `copy_` 等 adapter 冒充另一种语义。
3. 确认性能 Workload 的两阶段 Case API 和 `pytest_nodeid::level::dtype::ordinal` 标识规则可以长期维护，并接受其稳定性依赖已有 case 顺序不变；尚未迁移的旧式 Benchmark 继续正常全量运行，但不提供列举和重放能力。
4. 确认现有 case list、benchmark JSON 和 candidate call report 是否足够供自动化消费；如果需要单次 Preflight 返回全部 case 的成功/失败和阶段信息，应共同定义新的结果 Schema。同时确认结果字段 `candidate_source="override"` 是否保留，或改成能区分 `candidate`、`explicit_override` 和 `default` 的枚举。
5. 确认 FlagGems 只负责提供确定性的 candidate-only replay 和 CUDA profiler capture 区间，其他厂商 profiler 的启动、停止和结果解析仍由 KernelGen/外部工具负责；并使用 FlagGems 正式 CI 对迁移前后的相同用例做对比，确认默认路径没有非预期变化。

KernelGen 团队负责提供接口实现、测试迁移和沐曦机器上的回归结果；FlagGems 团队主要负责确认接口是否适合长期维护，以及通过正式 CI 判断改动能否合入。

## 是否合入的影响

### 如果不合入

**KernelGen 将长期维护一套只能在修改版 FlagGems 上运行的优化测试，优化阶段与最终提交阶段的评测可能不一致。**

KernelGen 后续编写的测试会统一依赖 `--candidate-code-path` 和直接算子解析链，但正式提交到 FlagGems 时仍要切换回主分支现有的调用方式。这意味着同一份算子代码需要经过两套路径验证，既增加适配和重复测试的工作量，也可能出现优化时通过、提交时失败，或者两边性能结论不同的情况。

此外，KernelGen 需要的 Workload 标识、单独重放和输入生成接口如果只存在于修改版 FlagGems 中，相关测试和工具便无法直接在主分支运行，后续每次同步或提交都需要额外改写。

### 如果合入

**整体风险可控，但不是零风险：部分现有测试需要接入统一的待测代码入口，调用路径变化可能改变原有测试结果。**

`--candidate-code-path`、`--list-cases`、`--case-id`、Preflight 和 Profile 等外部模式均为显式启用；不传这些参数时仍解析并运行默认 FlagGems 实现。但为了让同一份输入构造同时服务全量计时和单 case 重放，本次公共改动会让已支持的 Benchmark family 在默认执行时也经过两阶段 Case API，预期生成的输入序列和计时语义不变，仍需通过 FlagGems 正式 CI 验证。后续迁移到直接算子解析链的具体 pytest 风险更高：原有测试可能经过框架自动路由，改成直接调用函数后，参数处理和执行路径可能存在差异。沐曦机器的抽样对比已经观察到少量原本通过的用例在改为直接调用后失败，因此这项风险需要承认并通过分批回归控制。

建议只迁移 KernelGen 实际需要的测试，每一批迁移都对比修改前后的相同用例，并以 FlagGems 正式 CI 结果作为最终合入标准。FlagGems 后续需要维护的是少量公共接口和对应回归测试，而不是一套独立的 KernelGen 测试框架。

## 建议的合入方式

当前公共能力分支只包含下面前两层，具体算子的 pytest 迁移明确留到公共接口合入后的后续 PR。建议按依赖顺序评审和合入：

1. 公共 candidate 层：`candidate.py`、`pytest_plugin.py`、`resolve_gems_op()` 及其单元测试，选择少量代表性算子验证“不传参数时保持默认、传入参数时只运行指定待测代码”。
2. Benchmark Case API：Workload 数据结构、列举、重放、Preflight、Profile replay 及 `benchmark/test_benchmark_case_api.py`，验证列举阶段不创建 Tensor、重放与全量测试共用输入构造。
3. pytest 迁移（不属于本次分支）：最后分批迁移 KernelGen 必需的正确性和性能测试，每批都运行迁移前后同 case 对比和 FlagGems 正式 CI，避免一次性扩大影响范围。

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
3. 输入 Tensor 在 `flag_gems.device` 上生成，并使用 `accuracy_utils.to_reference()` 构造 reference 输入。必须先转换全部 reference 输入，再调用 PyTorch reference；不得先在目标设备执行 reference，再对计算结果调用 `to_reference()`。对于原地修改、out、alias 或可能修改输入的算子，reference 与待测代码必须使用内容相同但存储独立的输入，并同时检查返回值和被修改对象。
4. 传给直接函数的参数类型必须符合 FlagGems 公开 Python 接口。普通标量使用 Python 的 `int`、`float` 和 `bool`，不得依赖 dispatcher 把 NumPy 标量或其他特殊对象自动转换成 schema 类型。
5. 浮点结果使用 `gems_assert_close()`，整数、布尔值或必须逐位一致的结果使用 `gems_assert_equal()`。只有算子语义确有需要时才能调整容差，并在代码中说明原因；tuple、多输出、dtype、shape、mutation 和 alias 语义需要分别检查。
6. shape 和 dtype 优先复用 `accuracy_utils` 中的公共集合。新增本地 case 必须覆盖该算子的关键边界，而不是重复维护一套与正式测试不同的常规 shape。
7. 随机算子应固定测试所需的随机状态，并按算子语义选择确定性、边界或统计检查。不得通过放宽断言、重试直到通过或无理由 skip 来掩盖随机失败。
8. 厂商限制只能使用带明确原因的 `pytest.mark.skipif` 或 `xfail` 表达，reason 应关联已知问题或清楚说明硬件限制。不得因为当前待测代码失败而新增平台 skip。

#### `--ref cpu` 的输入转换顺序

`--ref cpu` 不是全局设备拦截器。它只在正确性测试中设置 `TO_CPU=True`，使 `accuracy_utils.to_reference()` 把传入的 Tensor 转到 CPU；Python 不会自动把任意 PyTorch reference 调用改到 CPU。因此测试代码必须先转换输入，再执行 reference。

下面的写法是错误的：Python 会先在目标设备执行内层算子，随后才把已经算出的结果移动到 CPU，无法绕过目标后端缺失的 PyTorch API。

```python
ref_out = utils.to_reference(
    torch.nn.functional.scaled_dot_product_attention(query, key, value)
)
```

正确写法是先转换全部 reference 输入：

```python
ref_query = utils.to_reference(query)
ref_key = utils.to_reference(key)
ref_value = utils.to_reference(value)

ref_out = torch.nn.functional.scaled_dot_product_attention(
    ref_query,
    ref_key,
    ref_value,
)
```

mask、index、weight、bias、out buffer 和其他 Tensor 参数也必须按同样规则转换。若同名底层 API 本身没有 CPU kernel，可以提供语义等价、独立且可审查的 PyTorch CPU reference；无法准确表达相同语义时，应明确标记 CPU reference 不支持，不得回到目标设备计算后再把结果搬到 CPU，也不得改变 workload、dtype 或参数语义来伪装成通过。

当前 Kernel Todo V2 的 84 个设备 Reference 不可用条目中，已经确认以下正确性测试存在 `--ref cpu` 接入问题：

| 问题类型 | 算子 | 当前行为 | 修改要求 |
| --- | --- | --- | --- |
| 先执行 reference，再调用 `to_reference()` | `_scaled_dot_product_efficient_attention`、`_scaled_dot_product_fused_attention_overrideable`、`_embedding_bag_dense_backward` | PyTorch reference 仍先在目标设备执行；这 3 类算子涉及摩尔线程、沐曦和昇腾的 8 个“芯片 × 算子”条目 | 先分别转换全部 reference 输入，再调用 PyTorch reference；不得只转换输出 |
| CPU 模式直接 skip | `rnn_relu` | `TO_CPU=True` 时不执行任何正确性 case | 如果 CPU 能表达相同 reference 语义，应只保留待测实现的真实平台限制，不应因 reference 位于 CPU 而跳过整个测试 |
| CPU reference 前依赖目标设备专用前置算子 | `cudnn_batch_norm_backward` | 在构造 CPU reference 前，先在目标设备调用 `cudnn_batch_norm` 生成统计量；目标后端缺少该 API 时仍会提前失败 | 使用能够在 CPU 独立构造的等价统计量或 reference 流程，确保 CPU reference 不依赖目标设备 reference API |
| 非 CUDA 设备直接 skip | `_scaled_dot_product_cudnn_attention` | 摩尔线程和昇腾不会执行正确性 case，即使 reference 可以使用 CPU 的通用 SDPA 表达 | 分开描述待测实现的平台能力与 reference 设备；无法在目标平台执行待测实现时明确记录 candidate 不支持，不要把它误记为 CPU reference 失败 |

这份清单记录的是当前已确认的问题，不是完整枚举。新增或迁移测试时，应检查 reference 调用发生在哪个设备、是否在 `to_reference()` 之前执行了任何设备算子，以及 `--ref cpu` 是否因平台 marker 变成全量 skip。

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
