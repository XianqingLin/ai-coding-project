# Agent 功能状态清单

> 本文档记录 ai-coding 项目 Agent 核心功能的当前状态、已完成项和待办项。
>
> 定位：个人学习项目，优先保障 Agent 能真正完成任务、方便快速迭代实验。

---

## P0 — 阻塞级（Agent 完成任务的基础能力）

### P0-1 修复 Agent 任务完成能力 ✅

**状态：已完成（2026-06-04）**

从"强制硬切换"重构为"软约束 + Agent 自主决策"模式：

1. **移除了 `MAX_EXPLORE_STEPS` 和硬阶段切换**
2. **引入 `plan` 工具** — Agent 自主提交修改计划，进入修改阶段
3. **引入软约束提醒** — 探索过多时递进式提醒（提示→要求→强制）
4. **引入 `set_todo` 工具** — 子任务跟踪

**验证结果**：

| 任务 | 状态 | 耗时 | 说明 |
|---|---|---|---|
| simple-calculator-bug | ✅ passed | 66.4s | 2 处修改，零重试 |
| simple-config-validation | ✅ passed | 69.7s | 3 文件跨依赖，零重试 |
| abs-module-cache-flags | ❌ timeout | 300s | 任务超纲（复杂 Go 解释器），Agent 能 plan 但无法在复杂场景下 edit |

---

### P0-2 实现 Mock LLM 模式 ✅

**状态：已完成**

- [x] `MockChatModel` 类（`src/ai_coding/mock_llm.py`）
- [x] 预设模式 + 交互模式
- [x] `create_lc_llm("mock")` 支持
- [x] `MOCK_INTERACTIVE` 配置

---

## P1 — 重要级

### P1-1 轨迹分析与调试工具

**状态：已完成核心部分（2026-06-04）**

`run_single.py` 已实现彩色终端输出和 trajectory JSON 保存。Agent 的 `run_with_trace()` 现在能正确捕获 `thinking` 事件（从 `reasoning_content` 提取）。

**解决方向**
- [x] `thinking` 事件捕获 — 通过 `KimiChatOpenAI._create_chat_result` 提取 `reasoning_content`
- [x] `--step` 单步调试模式 — 已实现
- [ ] `scripts/analyze_trace.py` — 统计耗时、工具频率、重复调用检测、循环模式检测（可选增强）

---

### P1-2 增强代码编辑工具 ✅

**状态：已完成核心迭代**

原始 `edit_file`（精确字符串替换）已被移除，替换为更可靠的 `str_replace_file`：

| 工具 | 状态 | 说明 |
|---|---|---|
| `str_replace_file` | ✅ | 字符串替换 + **唯一性检查** + 模糊匹配 fallback |
| `write_file` | ✅ | 全量覆盖 |
| `insert_after_line` | ✅ | 行后插入（保留，Agent 未使用过但功能可用） |
| `edit_file`（旧） | ❌ 已移除 | 精确替换，失败率高 |
| `view_diff` | ❌ 已移除 | Agent 未使用，非必要 |

**核心改进**：
- `str_replace_file` 要求 `old_string` 在文件中**唯一出现**
- 不唯一时报错并列出所有匹配位置
- 精确匹配失败时给出相似度最高的候选提示

---

## P2 — 优化级

### P2-1 轻量级单任务运行与调试 ✅

**状态：已完成**

`scripts/run_single.py` 已实现：
- [x] 单任务快速运行
- [x] `--step` 单步调试
- [x] `--prompt` 自定义系统提示
- [x] `--compare` trajectory 对比
- [x] Rich 彩色终端输出

---

### P2-2 系统提示与 Agent 参数配置化

**状态：未开始**

Agent 参数仍硬编码在 `langgraph_agent.py` 中：
- `max_iterations`
- 滑动窗口 `max_turns`
- 压缩阈值
- 系统提示路径

**解决方向**
- [ ] 提取到构造函数参数
- [ ] 支持 `--max-explore-steps --prompt` 等命令行覆盖
- [ ] `prompts/` 版本管理

---

### P2-3 Agent 中间状态可观测性

**状态：未开始**

**解决方向**
- [ ] `--dump-state` 导出 messages JSON
- [ ] TUI/日志中增加阶段标签、步数、token 数
- [ ] 轮次摘要文件

---

## 附录：已验证运行结果

| 任务 | 复杂度 | 状态 | 耗时 | 关键指标 |
|---|---|---|---|---|
| simple-calculator-bug | 低（1 文件，2 修改） | ✅ passed | 66.4s | plan→edit 链路打通，str_replace_file 零重试 |
| simple-config-validation | 中（3 文件，跨依赖） | ✅ passed | 69.7s | 多文件修改，一次成功 |
| abs-module-cache-flags | 高（Go 解释器） | ❌ timeout | 300s | 任务超纲，需增强 Agent 能力后再试 |

---

## 当前工具全览（9 个）

```
read_file, write_file, str_replace_file, insert_after_line,
list_dir, execute_command, grep, plan, set_todo
```

---

*最后更新：2026-06-04*
