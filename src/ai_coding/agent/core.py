"""LangGraph ReAct Agent 运行时容器.

AgentState 是自我管理容量的短期记忆容器，跨轮次保留.
- messages 由 LangGraph 的 add_messages reducer 累积，由 LangGraphAgent 管理容量
- file_snapshots 由 tools_node 动态更新，llm_node 调用前注入
- ContextCompressor 作为 AgentState 的容量管理工具，由 LangGraphAgent 显式调用
"""

import os
import time
import uuid
from pathlib import Path
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

from ai_coding.agent.context_compressor import ContextCompressor
from ai_coding.agent.nodes import (
    create_approval_gate,
    create_llm_node,
    create_should_continue,
    create_tools_node,
)
from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.prompts import PromptContext, SystemPromptBuilder
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
        prompt_context: Optional[PromptContext] = None,
        system_prompt_builder: Optional[SystemPromptBuilder] = None,
        thread_id: Optional[str] = None,
        enable_short_term_memory: bool = True,
        short_term_memory_budget: int = 100000,
        auto_approve: bool = False,
        work_dir: Optional[str] = None,
        llm_factory=None,
        on_state_change: Optional[Callable[[], None]] = None,
        on_wire_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_loop: Any = None,
        on_approval_request: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_edit_proposal: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm = llm
        self.llm_factory = llm_factory
        if llm_factory and not llm:
            self.llm = llm_factory()
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.prompt_context = prompt_context
        self.system_prompt = system_prompt or self._build_system_prompt(
            system_prompt_builder
        )
        self.thread_id = thread_id or uuid.uuid4().hex[:8]
        self.auto_approve = auto_approve
        self.work_dir = work_dir or str(Path.cwd())
        self.on_state_change = on_state_change
        self.on_wire_event = on_wire_event
        self.event_loop = event_loop
        self.on_approval_request = on_approval_request
        self.on_edit_proposal = on_edit_proposal

        # 流式输出时缓存 assistant 文本，用于 wire 记录
        self._pending_assistant_text: str = ""

        # 前端 approval 等待机制：按 request_id 索引多个 pending future
        self._approval_futures: Dict[str, Any] = {}

        # 流式运行取消标志
        self._stop_requested = False

        # 待处理的 edit_proposal（前端可 apply 或 reject）
        self._pending_edits: Dict[str, Dict[str, Any]] = {}

        # 唯一状态源：自我管理容量的短期记忆容器
        self.state: Optional[AgentState] = None

        # 依赖组装
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            # 向子 Agent 派发工具注入 LLM 依赖
            if hasattr(tool, "set_llm") and callable(getattr(tool, "set_llm")):
                tool.set_llm(self.llm, self.llm_factory)
            # 注入工作目录沙箱
            if hasattr(tool, "set_work_dir") and callable(
                getattr(tool, "set_work_dir")
            ):
                tool.set_work_dir(self.work_dir)
            # 子 Agent 工具继承父 Agent 的回调上下文
            if hasattr(tool, "set_parent_context") and callable(
                getattr(tool, "set_parent_context")
            ):
                tool.set_parent_context(
                    event_loop=self.event_loop,
                    on_approval_request=self.on_approval_request,
                    on_edit_proposal=self.on_edit_proposal,
                    work_dir=self.work_dir,
                )
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
        builder.add_node(
            "approval",
            create_approval_gate(
                tool_registry=self.tool_registry,
                interactive=not self.auto_approve,
                event_loop=self.event_loop,
                on_approval_request=self.on_approval_request,
                register_approval_future=self.register_approval_future,
            ),
        )
        builder.add_node(
            "tools",
            create_tools_node(
                tool_registry=self.tool_registry,
                on_edit_proposal=self._handle_edit_proposal,
            ),
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
            return {
                "messages": messages,
                "file_snapshots": {},
                "todos": [],
                "globally_approved_tools": [],
                "background_tasks": [],
                "plan_mode": False,
                "plan_file_path": "",
                "sub_agents": [],
            }

        # 复制现有历史并追加用户输入
        messages = list(self.state["messages"]) + [HumanMessage(content=user_input)]
        return {
            "messages": messages,
            "file_snapshots": dict(self.state.get("file_snapshots", {})),
            "todos": [dict(t) for t in self.state.get("todos", [])],
            "globally_approved_tools": list(
                self.state.get("globally_approved_tools", [])
            ),
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
                f"[Compact] 上下文超阈值({current_tokens}/"
                f"{self.context_compressor.token_budget}), 自动触发压缩"
            )
            self.compact()

    def _handle_edit_proposal(self, proposal: Dict[str, Any]) -> None:
        """内部处理 edit_proposal：缓存并透传给外部回调."""
        edit_id = proposal.get("id", "")
        if edit_id:
            self._pending_edits[edit_id] = proposal
        if self.on_edit_proposal is not None:
            try:
                self.on_edit_proposal(proposal)
            except Exception:
                logger.debug("edit_proposal 外部回调失败", exc_info=True)

    def apply_edit(self, edit_id: str) -> bool:
        """确认应用 edit_proposal.

        当前实现中文件已被工具写入，apply 仅表示用户确认保留。
        """
        if edit_id not in self._pending_edits:
            return False
        del self._pending_edits[edit_id]
        return True

    def reject_edit(self, edit_id: str) -> bool:
        """拒绝 edit_proposal，将文件恢复为旧内容.

        若旧内容为空且文件原本不存在，则删除文件。
        """
        if edit_id not in self._pending_edits:
            return False
        proposal = self._pending_edits.pop(edit_id)
        path = proposal.get("path", "")
        old_content = proposal.get("old_content", "")
        if not path:
            return False
        try:
            if old_content == "":
                # 原本就不存在，新写入后又被拒绝，删除文件
                if os.path.exists(path):
                    os.remove(path)
                return True
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(path, "w", encoding="utf-8") as f:
                f.write(old_content)
            return True
        except Exception as e:
            logger.error(f"恢复文件失败 {path}: {e}", exc_info=True)
            return False

    def register_approval_future(self, request_id: str, future: Any) -> Any:
        """注册一个等待前端响应的 approval future.

        若 request_id 已存在且未完成，则返回已有的 future（防止 LangGraph
        checkpoint 重放时重复创建）。注册成功后在 future 上附加
        done_callback，完成后自动清理 registry。

        Returns:
            应被等待的 future 对象（新注册的或已存在的）。
        """
        existing = self._approval_futures.get(request_id)
        if existing is not None and not existing.done():
            return existing

        self._approval_futures[request_id] = future

        def _cleanup(fut):
            # 仅当仍是当前注册的 future 时才清理，避免误删重放的注册
            if self._approval_futures.get(request_id) is fut:
                self._approval_futures.pop(request_id, None)

        future.add_done_callback(_cleanup)
        return future

    def request_stop(self) -> None:
        """请求取消当前正在运行的流式 Agent（协作式取消）."""
        self._stop_requested = True

    def submit_approval_response(self, request_id: str, choice: str) -> bool:
        """提交前端 approval 响应，解除对应 approval gate 阻塞.

        Args:
            request_id: 审批请求 ID（来自 approval_request 消息）.
            choice: "approve" / "reject" / "approve_all"

        Returns:
            是否成功提交（False 表示该 request_id 没有等待中的 approval）
        """
        future = self._approval_futures.pop(request_id, None)
        if future is None:
            return False
        future.set_result(choice)
        return True

    def _emit_wire_event(self, event: Dict[str, Any]) -> None:
        """发送交互事件到外部记录器（如 wire 持久化）.

        事件字典应至少包含 'type' 字段。若未注册回调则静默忽略。
        """
        if self.on_wire_event is None:
            return
        event_with_ts = {"timestamp": time.time(), **event}
        try:
            self.on_wire_event(event_with_ts)
        except Exception:
            logger.debug("wire 事件回调失败", exc_info=True)

    def _is_api_or_network_error(self, e: Exception) -> bool:
        """启发式判断异常是否为 API/网络相关."""
        module = type(e).__module__.lower()
        name = type(e).__name__.lower()
        indicators = [
            "openai",
            "anthropic",
            "google",
            "http",
            "urllib",
            "requests",
            "connectionerror",
            "timeout",
            "apierror",
            "ratelimit",
            "authenticationerror",
            "serviceunavailable",
        ]
        combined = f"{module}.{name}"
        err_str = str(e).lower()
        return any(ind in combined or ind in err_str for ind in indicators)

    def _handle_run_exception(self, e: Exception, method_name: str) -> str:
        """分类处理运行异常，返回用户友好的错误信息."""
        if isinstance(e, RecursionError):
            msg = (
                "Agent 迭代次数过多，已达到递归上限。"
                "请尝试简化任务或增加 max_iterations。"
            )
            logger.error(f"[{method_name}] 递归上限: {e}", exc_info=True)
        elif isinstance(e, InterruptedError):
            msg = "Agent 运行已取消。"
            logger.info(f"[{method_name}] 用户取消: {e}")
        elif self._is_api_or_network_error(e):
            msg = f"API/网络异常: {e}。请检查网络连接或 API 配置。"
            logger.error(f"[{method_name}] API/网络异常: {e}", exc_info=True)
        else:
            msg = f"Agent 执行失败: {e}"
            logger.error(f"[{method_name}] Agent 执行失败: {e}", exc_info=True)
        self._emit_wire_event({"type": "error", "text": msg})
        return msg

    def run(self, user_input: str) -> str:
        """处理用户输入（非流式）."""
        logger.info(f"用户输入: {user_input[:100]}")
        self._emit_wire_event({"type": "user_input", "input": user_input})
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
                self._emit_wire_event({"type": "error", "text": "Agent 未返回任何消息"})
                return "[错误] Agent 未返回任何消息."

            last_msg = messages[-1]
            content = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )
            if content:
                self._emit_wire_event({"type": "assistant", "text": content})
            logger.info(
                f"Agent 完成 | 消息数: {len(messages)} | 输出: {len(content)} 字符"
            )
            return content
        except Exception as e:
            msg = self._handle_run_exception(e, "run")
            return f"[错误] {msg}"

    def run_stream(self, user_input: str) -> Iterator[str]:
        """处理用户输入（流式输出）.

        流式输出结束后会通过 graph.get_state 获取最终状态并更新 self.state，
        从而支持跨轮次的历史累积和持久化.
        """
        logger.info(f"[流式] 用户输入: {user_input[:100]}")
        self._emit_wire_event({"type": "user_input", "input": user_input})
        self._pending_assistant_text = ""
        config = self._get_run_config()
        try:
            # 入口检查：必要时自动压缩
            self._maybe_compact()
            self._stop_requested = False

            for event in self.graph.stream(
                self._build_state(user_input),
                config=config,
                stream_mode="messages",
            ):
                if self._stop_requested:
                    raise InterruptedError("Agent run cancelled by client")
                msg, metadata = event
                if metadata.get("langgraph_node") == "llm":
                    content = msg.content if hasattr(msg, "content") else ""
                    if content:
                        self._pending_assistant_text += content
                        yield content

            # 流式结束后获取最终状态并持久化
            try:
                final_snapshot = self.graph.get_state(config)
                if final_snapshot is not None and hasattr(final_snapshot, "values"):
                    self.state = final_snapshot.values
                elif isinstance(final_snapshot, dict):
                    self.state = final_snapshot
                if self._pending_assistant_text:
                    self._emit_wire_event(
                        {"type": "assistant", "text": self._pending_assistant_text}
                    )
                    self._pending_assistant_text = ""
                if self.on_state_change:
                    try:
                        self.on_state_change()
                    except Exception:
                        pass
                logger.info("[流式] 最终状态已保存")
            except Exception as e:
                logger.warning(f"[流式] 获取最终状态失败: {e}")
        except Exception as e:
            msg = self._handle_run_exception(e, "run_stream")
            yield f"[错误] {msg}"
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
        self._emit_wire_event({"type": "user_input", "input": user_input})
        self._pending_assistant_text = ""
        config = self._get_run_config()
        current_phase: Optional[str] = None  # None, "thinking", "assistant"

        try:
            self._maybe_compact()
            self._stop_requested = False

            for event in self.graph.stream(
                self._build_state(user_input),
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if self._stop_requested:
                    raise InterruptedError("Agent run cancelled by client")
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
                        self._pending_assistant_text += content
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
                                        tool_name = tc.get("name", "")
                                        tool_args = tc.get("args", {})
                                        self._emit_wire_event(
                                            {
                                                "type": "tool_call",
                                                "name": tool_name,
                                                "args": tool_args,
                                            }
                                        )
                                        yield {
                                            "type": "tool_call",
                                            "name": tool_name,
                                            "args": tool_args,
                                        }
                        elif node_name == "tools":
                            messages = update.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, ToolMessage):
                                    text = msg.content or ""
                                    self._emit_wire_event(
                                        {"type": "observation", "text": text}
                                    )
                                    yield {"type": "observation", "text": text}

            # 结束当前阶段
            if current_phase == "thinking":
                yield {"type": "thinking_end"}
            elif current_phase == "assistant":
                yield {"type": "assistant_end"}

            if self._pending_assistant_text:
                self._emit_wire_event(
                    {"type": "assistant", "text": self._pending_assistant_text}
                )
                self._pending_assistant_text = ""

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
            msg = self._handle_run_exception(e, "run_stream_verbose")
            self._emit_wire_event({"type": "error", "text": msg})
            yield {"type": "error", "text": msg}

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
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content or "",
                        "tool_call_id": msg.tool_call_id,
                    }
                )
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content or ""})
        return result

    def _build_system_prompt(
        self, builder: Optional[SystemPromptBuilder] = None
    ) -> str:
        """使用 SystemPromptBuilder 构建 system prompt."""
        if builder is None:
            builder = SystemPromptBuilder()
        return builder.build(self.prompt_context)

    def get_context_usage(self) -> dict:
        """计算当前会话的上下文窗口使用率.

        基于 self.state["messages"] 中的实际消息（已被管理）+ file_snapshots.
        """
        if tiktoken is None:
            return {
                "used_tokens": 0,
                "limit_tokens": DEFAULT_CONTEXT_LIMIT,
                "percentage": 0.0,
            }
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {
                "used_tokens": 0,
                "limit_tokens": DEFAULT_CONTEXT_LIMIT,
                "percentage": 0.0,
            }

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
