# AI Coding Assistant — 简历项目介绍

---

参考 Claude Code 实现的控制台 AI 编程助手，基于 **LangGraph ReAct 状态机** 协调大模型推理、工具调用与用户审批，封装 **Kimi / OpenAI / Mock** 三种 LLM Provider，支持 **Function Calling、流式输出与 `reasoning_content` 解析**，实现自然语言驱动的代码阅读、编辑、Shell 执行与多步任务规划。

核心工作：

● 构建 **结构化工具系统**：抽象 `Tool` 基类与 `ToolRegistry`，内置 24 个工具覆盖文件读写、代码搜索、Git 操作、Shell 执行、后台任务、子 Agent 委派；通过工作目录边界校验、命令静态安全校验与交互式审批门控三层机制约束写操作、Shell 执行等敏感行为。

● 设计 **Agent 记忆机制**：已实现上下文压缩与 `AgentState` 序列化持久化；长期记忆链路已完成设计，计划由 LLM 自动提取用户偏好与项目规则，用于优化后续会话的系统提示。

● 实现 **多会话状态管理**：支持会话创建、切换、重命名、删除与状态恢复，控制长对话 Token 开销，避免超出模型上下文窗口。

● 工程规范：**380+** 单元测试、**85%** 行覆盖率，`black / isort / flake8 / mypy` 全过，GitHub Actions CI 在 Python 3.10/3.11/3.12 全绿。

---

*最后更新：2026-06-19*
