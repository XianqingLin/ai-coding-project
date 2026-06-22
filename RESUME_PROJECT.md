# AI Coding

基于 LangGraph ReAct 状态机的控制台 AI 编程助手，参考 Claude Code 设计，支持自然语言驱动的代码阅读、编辑、Shell 执行与多步任务规划。

核心工作：

● 基于 **LangGraph ReAct 状态机**实现 Agent 推理与工具调用循环：拆分 LLM 节点、工具节点、审批节点与边路由；封装 **Kimi / OpenAI / Mock** 三种 LLM Provider，支持 Function Calling、流式输出与 `reasoning_content` 解析，并在工具调用前后加入交互式审批门控。

● 构建 **结构化工具系统**：抽象 `Tool` 基类与 `ToolRegistry`，内置 24 个工具覆盖文件读写、代码搜索、Git 操作、Shell 执行、后台任务、子 Agent 委派；统一封装 `ToolResult` 并自动生成 OpenAI Function Calling / LangChain StructuredTool Schema；通过工作目录边界校验、命令静态安全校验与交互式审批门控约束写操作、Shell 执行等敏感行为。

● 设计并实现 **Agent 记忆机制**：短期记忆通过上下文压缩组件对对话历史进行 token 预算控制，对旧轮次去重、摘要与截断，避免超出模型上下文窗口，并在 token 用量超过 95% 阈值时自动触发压缩；同时将会话状态序列化持久化到本地，支持会话中断后恢复。TUI 中的 `/compact` 命令主要用于手动压缩当前会话上下文、释放窗口，可附带焦点指令（如 `/compact 保留数据库设计`）以在摘要中保留关键信息；压缩过程中顺手由大模型从原始对话中提取用户偏好、项目规则与关键事实作为长期记忆，以结构化条目按用户级与项目级分层持久化（用户级可在不同项目间共享）。新会话启动时检索相关记忆并注入系统提示的“记忆摘要”段落，实现跨会话的个性化上下文增强。

● 实现 **多会话状态管理与交互层**：支持会话创建、切换、重命名、删除与完整状态恢复；提供 CLI 与 Textual 全屏 TUI 入口，控制长对话 Token 开销。工程上保持 **435** 单元测试、**86%** 行覆盖率，`black / isort / flake8 / mypy` 全过，GitHub Actions CI 在 Python 3.10/3.11/3.12 全绿。

---

*最后更新：2026-06-22*
