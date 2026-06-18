# AI Coding Benchmark

可量化、可复现的 Agent benchmark 系统。

## 设计原则

- **客观判题**：每个任务配一个可执行的判题脚本（pytest / python / shell），以 exit code 决定 pass/fail。
- **隔离运行**：每个任务在独立临时目录中运行，避免互相污染。
- **量化指标**：自动收集耗时、轮次、工具调用次数、消息数、token 用量。
- **多轮反馈**：任务可在 `max_rounds` 内根据测试输出自迭代修复。

## 目录结构

```text
benchmarks/
├── runner.py              # Benchmark 运行器
├── loader.py              # 任务加载器
├── spec.py                # 数据模型
├── reporter.py            # 报告输出
├── __main__.py            # CLI 入口
├── tasks/                 # 任务集
│   ├── tetris/
│   ├── fibonacci/
│   └── arch_stress/       # 架构压力任务集
└── reports/               # 输出报告
```

## 任务配置 `task.json`

```json
{
  "name": "fibonacci",
  "description": "实现斐波那契数函数并通过单元测试",
  "prompt": "请在当前目录下创建 fib.py，实现函数 fib(n)。",
  "max_rounds": 3,
  "auto_approve": true,
  "initial_files": ["tests/test_fib.py"],
  "evaluation": {
    "type": "script",
    "command": "python tests/test_fib.py",
    "timeout": 30,
    "expected_exit_code": 0
  }
}
```

字段说明：

- `prompt`: 发给 Agent 的初始任务描述。
- `max_rounds`: 最多运行几轮（第一轮为 prompt，后续为测试反馈）。
- `initial_files`: 需要预先复制到工作目录的文件/目录，支持 glob，保持相对目录结构。
- `evaluation.command`: 判题命令，在工作目录下执行。
- `evaluation.expected_exit_code`: 通过时的期望退出码。

## 使用方式

```bash
# 跑全部任务
python -m benchmarks --tasks benchmarks/tasks --model kimi

# 跑单个任务
python -m benchmarks --tasks benchmarks/tasks/fibonacci --model openai

# 用 mock_llm 做离线回归测试
python -m benchmarks --tasks benchmarks/tasks/fibonacci --model mock

# 保留临时目录用于调试
python -m benchmarks --tasks benchmarks/tasks/tetris --keep-work-dir
```

退出码：所有任务通过时返回 `0`，否则返回 `1`，可用于 CI。

## 架构压力任务集

`benchmarks/tasks/arch_stress/` 包含 5 个专门考察 Agent 架构能力的任务：

- `rename_function`：多文件重构
- `fix_bug_iterative`：测试驱动迭代修复
- `long_file_edit`：长文件精准定位修改
- `multi_file_read`：读取多个文件并生成输出
- `add_feature`：增量开发，不破坏已有功能

```bash
python -m benchmarks --tasks benchmarks/tasks/arch_stress --model kimi
```

详见 `benchmarks/tasks/arch_stress/README.md`。

## 输出报告

报告为 JSON，包含 `metadata`、`summary` 和每个任务的详细结果：

```json
{
  "metadata": {"model": "kimi", "total_tasks": 2},
  "summary": {
    "total_tasks": 2,
    "passed": 1,
    "failed": 1,
    "pass_rate": 0.5,
    "avg_duration": 45.2,
    "avg_rounds": 1.5,
    "avg_tool_calls": 5.0,
    "avg_token_usage": 8234
  },
  "tasks": [...]
}
```

## 添加新任务

1. 在 `benchmarks/tasks/` 下新建目录。
2. 编写 `task.json` 和判题脚本（建议放在 `tests/` 或 `eval/` 下）。
3. 如需初始文件，在 `initial_files` 中列出相对路径。
4. 运行 `python -m benchmarks --tasks benchmarks/tasks/<your_task> --model mock` 做快速冒烟测试。
