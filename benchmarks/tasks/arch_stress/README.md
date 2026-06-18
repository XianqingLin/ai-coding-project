# Architecture Stress Tasks

这组任务专门用来评估 Agent 的**架构能力**，而不是单纯的函数级代码生成。

## 任务列表

| 任务 | 考察能力 | 说明 |
|---|---|---|
| `rename_function` | 多文件重构 | 重命名 `utils.py` 中的函数，并同步更新 `main.py` 的 import 和调用 |
| `fix_bug_iterative` | 测试驱动修复 | `calculator.py` 有 bug，需要先跑测试、读输出、再修复 |
| `long_file_edit` | 长文件精准编辑 | 在 20+ 个占位函数中定位 `process_data` 并修改 |
| `multi_file_read` | 多文件读取与生成 | 读 `config.json` + `template.txt`，生成 `output.txt` |
| `add_feature` | 增量开发 | 在不破坏现有 `reverse` 函数的前提下新增 `is_palindrome` |

## 运行

```bash
# 跑全部架构压力任务
python -m benchmarks --tasks benchmarks/tasks/arch_stress --model kimi

# 跑单个任务
python -m benchmarks --tasks benchmarks/tasks/arch_stress/rename_function --model kimi
```
