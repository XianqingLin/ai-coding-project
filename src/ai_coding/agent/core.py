"""LangGraph ReAct Agent 运行时容器.

彻底合并后：AgentState 是自我管理容量的短期记忆容器，跨轮次保留.
- messages 由 LangGraph 的 add_messages reducer 累积，由 LangGraphAgent 管理容量
- file_snapshots 由 tools_node 动态更新，llm_node 调用前注入
- ShortTermMemory 作为 AgentState 的容量管理工具，由 LangGraphAgent 显式调用
"""

import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ai_coding.agent.nodes import (
    create_approval_gate,
    create_llm_node,
    create_should_continue,
    create_tools_node,
)
from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.agent.context_compressor import ContextCompressor
from ai_coding.tools.base import Tool, ToolRegistry

logger = get_logger(__name__)

# 默认上下文窗口上限 (tokens)
DEFAULT_CONTEXT_LIMIT = 128000

# 自动压缩触发阈值（相对于 token_budget 的比例）
AUTO_COMPACT_THRESHOLD = 0.95


class LangGraphAgent:
    """基于 LangGraph 的 ReAct Agent 运行时容器.

    职责边界：
    - 组装 LLM、工具、上下文管理器等依赖
    - 构建并编译 StateGraph
    - 对外暴露运行接口（run / run_stream / run_stream_verbose / compact）
    - 通过 self.state 跨轮次管理对话历史和文件快照
    - 管理 AgentState 的容量（显式/自动压缩）
    """

    def __init__(
        self,
        llm=None,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        streaming: bool = True,
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        enable_short_term_memory: bool = True,
        short_term_memory_budget: int = 100000,
        auto_approve: bool = False,
        llm_factory=None,
        on_state_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.llm = llm
        self.llm_factory = llm_factory
        if llm_factory and not llm:
            self.llm = llm_factory()
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.system_prompt = system_prompt or self._build_system_prompt()
        self.thread_id = thread_id or uuid.uuid4().hex[:8]
        self.auto_approve = auto_approve
        self.on_state_change = on_state_change

        # 唯一状态源：自我管理容量的短期记忆容器
        self.state: Optional[AgentState] = None

        # 依赖组装
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            # 向子 Agent 派发工具注入 LLM 依赖
            if hasattr(tool, "set_llm") and callable(getattr(tool, "set_llm")):
                tool.set_llm(self.llm, self.llm_factory)
            self.tool_registry.register(tool)

        # 容量管理工具：用于 compact 和 _maybe_compact
        self.context_compressor = (
            ContextCompressor(token_budget=short_term_memory_budget)
            if enable_short_term_memory
            else None
        )

        # 构建图
        self.graph = self._build_graph()

        logger.info(
            f"LangGraph Agent 初始化完成 | 工具: {[t.name for t in self.tools]} | "
            f"流式: {streaming} | 会话: {self.thread_id}"
        )

    def _build_graph(self):
        """构建 ReAct 图结构：依赖注入节点工厂函数."""
        all_lc_tools = [t.to_langchain_tool() for t in self.tools]
        bound_llm = self.llm.bind_tools(all_lc_tools)

        builder = StateGraph(AgentState)
        builder.add_node("llm", create_llm_node(llm=bound_llm))
        builder.add_node("approval", create_approval_gate(
            tool_registry=self.tool_registry,
            interactive=not self.auto_approve,
        ))
        builder.add_node(
            "tools",
            create_tools_node(tool_registry=self.tool_registry),
        )

        # 图拓扑：START -> LLM -> (条件) -> approval -> 工具 -> 回到 LLM
        builder.add_edge(START, "llm")
        builder.add_conditional_edges(
            "llm",
            create_should_continue(
                tool_registry=self.tool_registry,
                auto_approve=self.auto_approve,
            ),
            {"approval": "approval", "tools": "tools", END: END},
        )
        builder.add_edge("approval", "tools")
        builder.add_edge("tools", "llm")

        return builder.compile(checkpointer=MemorySaver())

    def _get_run_config(self) -> dict:
        """构建运行配置（recursion_limit 用于防止无限循环）."""
        return {
            "recursion_limit": self.max_iterations * 5 + 20,
            "configurable": {"thread_id": self.thread_id},
        }

    def _build_state(self, user_input: str) -> dict:
        """构建本轮的初始 State.

        如果是第一轮，组装 [System, Human] 和空 file_snapshots.
        如果是后续轮，将现有历史与新用户输入一起传入.
        """
        if self.state is None:
            messages: List[BaseMessage] = []
            if self.system_prompt:
                messages.append(SystemMessage(content=self.system_prompt))
            messages.append(HumanMessage(content=user_input))
            return {"messages": messages, "file_snapshots": {}, "todos": [], "globally_approved_tools": [], "background_tasks": [], "plan_mode": False, "plan_file_path": "", "sub_agents": []}

        # 复制现有历史并追加用户输入
        messages = list(self.state["messages"]) + [HumanMessage(content=user_input)]
        return {
            "messages": messages,
            "file_snapshots": dict(self.state.get("file_snapshots", {})),
            "todos": [dict(t) for t in self.state.get("todos", [])],
            "globally_approved_tools": list(self.state.get("globally_approved_tools", [])),
            "background_tasks": list(self.state.get("background_tasks", [])),
            "plan_mode": bool(self.state.get("plan_mode", False)),
            "plan_file_path": str(self.state.get("plan_file_path", "")),
            "sub_agents": list(self.state.get("sub_agents", [])),
        }

    def compact(self) -> str:
        """显式压缩 AgentState，释放上下文空间.

        压缩结果直接替换 state["messages"] 中的原始消息.
        """
        if self.state is None or not self.context_compressor:
            return "当前无需压缩"

        original_count = len(self.state["messages"])
        original_tokens = self.context_compressor._estimate_tokens(
            list(self.state["messages"])
        )

        compressed = self.context_compressor.compress(list(self.state["messages"]))
        self.state["messages"] = compressed

        new_tokens = self.context_compressor._estimate_tokens(compressed)
        msg = (
            f"上下文已压缩: {original_count}条/{original_tokens}token → "
            f"{len(compressed)}条/{new_tokens}token"
        )
        logger.info(f"[Compact] {msg}")
        return msg

    def _maybe_compact(self) -> None:
        """检查当前上下文用量，超过阈值时自动压缩."""
        if self.state is None or not self.context_compressor:
            return

        current_tokens = self.context_compressor._estimate_tokens(
            list(self.state["messages"])
        )
        threshold = self.context_compressor.token_budget * AUTO_COMPACT_THRESHOLD

        if current_tokens > threshold:
            logger.warning(
                f"[Compact] 上下文超阈值({current_tokens}/{self.context_compressor.token_budget}), "
                f"自动触发压缩"
            )
            self.compact()

    def run(self, user_input: str) -> str:
        """处理用户输入（非流式）."""
        logger.info(f"用户输入: {user_input[:100]}")
        try:
            # 入口检查：必要时自动压缩
            self._maybe_compact()

            initial = self._build_state(user_input)
            result = self.graph.invoke(
                initial,
                config=self._get_run_config(),
            )
            # 保存最终 state（跨轮次保留）
            self.state = result
            if self.on_state_change:
                try:
                    self.on_state_change()
                except Exception:
                    pass

            messages = list(result.get("messages", []))
            if not messages:
                return "[错误] Agent 未返回任何消息."

            last_msg = messages[-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            logger.info(f"Agent 完成 | 消息数: {len(messages)} | 输出: {len(content)} 字符")
            return content
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            return f"[错误] Agent 执行失败: {e}"

    def run_stream(self, user_input: str) -> Iterator[str]:
        """处理用户输入（流式输出）.

        流式输出结束后会通过 graph.get_state 获取最终状态并更新 self.state，
        从而支持跨轮次的历史累积和持久化.
        """
        logger.info(f"[流式] 用户输入: {user_input[:100]}")
        config = self._get_run_config()
        try:
            # 入口检查：必要时自动压缩
            self._maybe_compact()

            for event in self.graph.stream(
                self._build_state(user_input),
                config=config,
                stream_mode="messages",
            ):
                msg, metadata = event
                if metadata.get("langgraph_node") == "llm":
                    content = msg.content if hasattr(msg, "content") else ""
                    if content:
                        yield content

            # 流式结束后获取最终状态并持久化
            try:
                final_snapshot = self.graph.get_state(config)
                if final_snapshot is not None and hasattr(final_snapshot, "values"):
                    self.state = final_snapshot.values
                elif isinstance(final_snapshot, dict):
                    self.state = final_snapshot
                if self.on_state_change:
                    try:
                        self.on_state_change()
                    except Exception:
                        pass
                logger.info("[流式] 最终状态已保存")
            except Exception as e:
                logger.warning(f"[流式] 获取最终状态失败: {e}")
        except Exception as e:
            logger.error(f"[流式] Agent 执行失败: {e}", exc_info=True)
            yield f"[错误] Agent 执行失败: {e}"
            return

    def run_stream_verbose(self, user_input: str) -> Iterator[Dict[str, Any]]:
        """流式+verbose 模式.

        同时输出：
        - thinking（reasoning_content）流式显示
        - assistant（content）流式显示
        - tool_call 一次性显示
        - observation 一次性显示

        使用 stream_mode=["messages", "updates"] 同时监听流式消息和节点更新.
        """
        logger.info(f"[流式+verbose] 用户输入: {user_input[:100]}")
        config = self._get_run_config()
        current_phase: Optional[str] = None  # None, "thinking", "assistant"

        try:
            self._maybe_compact()

            for event in self.graph.stream(
                self._build_state(user_input),
                config=config,
                stream_mode=["messages", "updates"],
            ):
                mode, data = event

                if mode == "messages":
                    msg, metadata = data
                    node = metadata.get("langgraph_node") if metadata else None
                    if node != "llm":
                        continue

                    reasoning = ""
                    content = ""

                    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                        reasoning = msg.additional_kwargs.get("reasoning_content") or ""

                    if hasattr(msg, "content") and msg.content:
                        content = msg.content

                    # thinking chunk
                    if reasoning:
                        if current_phase != "thinking":
                            if current_phase == "assistant":
                                yield {"type": "assistant_end"}
                            current_phase = "thinking"
                            yield {"type": "thinking_start"}
                        yield {"type": "thinking_chunk", "text": reasoning}

                    # assistant chunk
                    if content:
                        if current_phase == "thinking":
                            yield {"type": "thinking_end"}
                            current_phase = "assistant"
                            yield {"type": "assistant_start"}
                        elif current_phase != "assistant":
                            current_phase = "assistant"
                            yield {"type": "assistant_start"}
                        yield {"type": "assistant_chunk", "text": content}

                elif mode == "updates":
                    for node_name, update in data.items():
                        if node_name == "llm":
                            messages = update.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        yield {
                                            "type": "tool_call",
                                            "name": tc.get("name", ""),
                                            "args": tc.get("args", {}),
                                        }
                        elif node_name == "tools":
                            messages = update.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, ToolMessage):
                                    text = msg.content or ""
                                    yield {"type": "observation", "text": text}

            # 结束当前阶段
            if current_phase == "thinking":
                yield {"type": "thinking_end"}
            elif current_phase == "assistant":
                yield {"type": "assistant_end"}

            # 保存最终状态
            try:
                final_snapshot = self.graph.get_state(config)
                if final_snapshot is not None and hasattr(final_snapshot, "values"):
                    self.state = final_snapshot.values
                elif isinstance(final_snapshot, dict):
                    self.state = final_snapshot
                if self.on_state_change:
                    try:
                        self.on_state_change()
                    except Exception:
                        pass
                logger.info("[流式+verbose] 最终状态已保存")
            except Exception as e:
                logger.warning(f"[流式+verbose] 获取最终状态失败: {e}")

        except Exception as e:
            logger.error(f"[流式+verbose] Agent 执行失败: {e}", exc_info=True)
            yield {"type": "error", "text": f"Agent error: {e}"}

    def get_history(self) -> List[Dict[str, Any]]:
        """获取当前会话的完整对话历史."""
        if self.state is None:
            return []
        return self._messages_to_dicts(list(self.state.get("messages", [])))

    def clear_history(self) -> None:
        """清空对话历史（开启新会话）."""
        old_id = self.thread_id
        self.state = None
        self.thread_id = uuid.uuid4().hex[:8]
        logger.info(f"已开启新会话 | 旧: {old_id} -> 新: {self.thread_id}")

    def _messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """将 LangChain 消息列表转换为字典列表."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content or ""})
            elif isinstance(msg, AIMessage):
                d = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("args", {}),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "content": msg.content or "",
                    "tool_call_id": msg.tool_call_id,
                })
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content or ""})
        return result

    def _build_system_prompt(self) -> str:
        """从文件读取默认系统提示模板，并动态插入工具描述."""
        from pathlib import Path

        prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "default_system_prompt.txt"
        template = prompt_path.read_text(encoding="utf-8")

        tool_descriptions = []
        for tool in self.tools:
            params = ", ".join(p.name for p in tool.parameters)
            tool_descriptions.append(f"  - {tool.name}({params}): {tool.description}")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "  (暂无可用工具)"

        return template.format(tools_text=tools_text)

    def get_context_usage(self) -> dict:
        """计算当前会话的上下文窗口使用率.

        基于 self.state["messages"] 中的实际消息（已被管理）+ file_snapshots.
        """
        if tiktoken is None:
            return {"used_tokens": 0, "limit_tokens": DEFAULT_CONTEXT_LIMIT, "percentage": 0.0}
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {"used_tokens": 0, "limit_tokens": DEFAULT_CONTEXT_LIMIT, "percentage": 0.0}

        total_tokens = 0

        # 系统提示
        if self.system_prompt:
            total_tokens += len(encoder.encode(self.system_prompt))

        # state 中的消息（已被管理，反映实际注入 LLM 的体积）
        if self.state:
            for msg in self.state.get("messages", []):
                content = msg.content if hasattr(msg, "content") else ""
                if content:
                    total_tokens += len(encoder.encode(content))

            # file_snapshots
            for path, content in self.state.get("file_snapshots", {}).items():
                total_tokens += len(encoder.encode(content))

        limit = DEFAULT_CONTEXT_LIMIT
        percentage = round(total_tokens / limit * 100, 1)
        return {
            "used_tokens": total_tokens,
            "limit_tokens": limit,
            "percentage": percentage,
        }

    def get_stats(self) -> dict:
        """获取 Agent 运行统计."""
        msg_count = 0
        if self.state:
            msg_count = len(self.state.get("messages", []))
        return {
            "message_count": msg_count,
            "tool_count": len(self.tools),
            "tools": [t.name for t in self.tools],
            "streaming": self.streaming,
            "type": "langgraph",
            "thread_id": self.thread_id,
        }
