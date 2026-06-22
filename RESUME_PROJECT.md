# AI Coding

基于 LangGraph ReAct 状态机的控制台 AI 编程助手，参考 Claude Code 设计，支持自然语言驱动的代码阅读、编辑、Shell 执行与多步任务规划。

核心工作：

● 基于 **LangGraph ReAct 状态机**实现 Agent 推理与工具调用循环：拆分 LLM 节点、工具节点、审批节点与边路由；封装 **Kimi / OpenAI / Mock** 三种 LLM Provider，支持 Function Calling、流式输出与 `reasoning_content` 解析，并在工具调用前后加入交互式审批门控。

● 构建 **结构化工具系统**：抽象 `Tool` 基类与 `ToolRegistry`，内置 24 个工具覆盖文件读写、代码搜索、Git 操作、Shell 执行、后台任务、子 Agent 委派；统一封装 `ToolResult` 并自动生成 OpenAI Function Calling / LangChain StructuredTool Schema；通过工作目录边界校验、命令静态安全校验与交互式审批门控约束写操作、Shell 执行等敏感行为。

● 设计并实现 **Agent 记忆机制**：短期记忆由 `ContextCompressor` 对对话历史做 token 预算控制，对旧轮次去重、摘要与截断，避免超出模型上下文窗口；`AgentState` 序列化持久化到 `data/sessions/`，支持会话状态恢复；长期记忆通过 TUI `/compact` 命令由 LLM 提取用户偏好、项目规则与关键事实，以结构化 `MemoryEntry` 按用户级 / 项目级分层存储（用户级跨项目共享），新会话启动时检索相关记忆并注入 `SystemPromptBuilder` 的 `# 记忆摘要` 段落，实现跨会话的个性化上下文增强。

● 实现 **多会话状态管理与交互层**：支持会话创建、切换、重命名、删除与完整状态恢复；提供 CLI 与 Textual 全屏 TUI 入口，控制长对话 Token 开销。工程上保持 **435** 单元测试、**86%** 行覆盖率，`black / isort / flake8 / mypy` 全过，GitHub Actions CI 在 Python 3.10/3.11/3.12 全绿。

---

*最后更新：2026-06-22*
