"""LangGraph ReAct Agent 运行时容器.

负责依赖组装、图编译、运行入口暴露.
节点实现已抽离到 agent.nodes 子包，上下文管理抽离到 agent.context，
阶段控制抽离到 agent.phase.
"""

import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

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
from langgraph.graph import END, MessagesState, START, StateGraph

from ai_coding.agent.context import ContextManager
from ai_coding.agent.nodes import create_agent_node, create_should_continue, create_tools_node
from ai_coding.agent.phase import PhaseController
from ai_coding.logger import get_logger
from ai_coding.tools.base import Tool, ToolRegistry

logger = get_logger(__name__)


class LangGraphAgent:
    """基于 LangGraph 的 ReAct Agent 运行时容器.

    职责边界：
    - 组装 LLM、工具、上下文管理器、阶段控制器等依赖
    - 构建并编译 StateGraph
    - 对外暴露运行接口（run / run_stream / run_with_trace）
    - 管理 thread_id 和 checkpointer

    不直接实现节点业务逻辑，节点通过 agent.nodes 工厂函数注入.
    """

    def __init__(
        self,
        llm,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        streaming: bool = True,
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        checkpointer=None,
    ) -> None:
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.system_prompt = system_prompt or self._build_system_prompt()
        self.thread_id = thread_id or uuid.uuid4().hex[:8]
        self.checkpointer = checkpointer or MemorySaver()

        # 依赖组装
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            self.tool_registry.register(tool)

        self.phase_ctrl = PhaseController()
        self.context_mgr = ContextManager(system_prompt=self.system_prompt)

        # 构建图
        self.agent = self._build_graph()

        logger.info(
            f"LangGraph Agent 初始化完成 | 工具: {[t.name for t in self.tools]} | "
            f"流式: {streaming} | 会话: {self.thread_id}"
        )

    def _build_graph(self):
        """构建 ReAct 图结构：依赖注入节点工厂函数."""
        all_lc_tools = [t.to_langchain_tool() for t in self.tools]
        bound_llm = self.llm.bind_tools(all_lc_tools)

        builder = StateGraph(MessagesState)
        builder.add_node(
            "agent",
            create_agent_node(
                llm=bound_llm,
                context_mgr=self.context_mgr,
                phase_ctrl=self.phase_ctrl,
            ),
        )
        builder.add_node(
            "tools",
            create_tools_node(
                tool_registry=self.tool_registry,
                phase_ctrl=self.phase_ctrl,
            ),
        )
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", create_should_continue(), {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=self.checkpointer)

    def _get_run_config(self) -> dict:
        """构建运行配置（包含 thread_id 和 recursion_limit）."""
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.max_iterations * 5 + 20,
        }

    def _build_input(self, user_input: str) -> dict:
        """构建 Agent 输入消息."""
        return {"messages": [HumanMessage(content=user_input)]}

    def run(self, user_input: str) -> str:
        """处理用户输入（非流式）."""
        logger.info(f"用户输入: {user_input[:100]}")
        try:
            result = self.agent.invoke(
                self._build_input(user_input),
                config=self._get_run_config(),
            )
            messages = result.get("messages", [])
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
        """处理用户输入（流式输出）."""
        logger.info(f"[流式] 用户输入: {user_input[:100]}")
        try:
            for event in self.agent.stream(
                self._build_input(user_input),
                config=self._get_run_config(),
                stream_mode="messages",
            ):
                msg, metadata = event
                if metadata.get("langgraph_node") == "agent":
                    content = msg.content if hasattr(msg, "content") else ""
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"[流式] Agent 执行失败: {e}", exc_info=True)
            yield f"[错误] Agent 执行失败: {e}"

    def run_with_trace(self, user_input: str) -> Iterator[Dict[str, Any]]:
        """处理用户输入，返回完整的工作流程轨迹（用于 verbose 显示）."""
        logger.info(f"[轨迹] 用户输入: {user_input[:100]}")
        start_time = time.time()
        step = 0
        try:
            for chunk in self.agent.stream(
                self._build_input(user_input),
                config=self._get_run_config(),
                stream_mode="updates",
            ):
                for node, data in chunk.items():
                    ts = round(time.time() - start_time, 2)
                    if node == "agent":
                        msg = data["messages"][0]
                        if isinstance(msg, AIMessage):
                            step += 1
                            if msg.tool_calls:
                                thinking_text = msg.additional_kwargs.get("reasoning_content") or msg.content
                                if thinking_text:
                                    event = {"type": "thinking", "text": thinking_text, "timestamp": ts, "step": step}
                                    logger.info(f"[轨迹] step={step} thinking_len={len(thinking_text)}")
                                    yield event
                                for tc in msg.tool_calls:
                                    event = {
                                        "type": "tool_call",
                                        "name": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                        "id": tc.get("id", ""),
                                        "timestamp": ts,
                                        "step": step,
                                    }
                                    logger.info(f"[轨迹] step={step} tool_call={tc.get('name', '')} args={tc.get('args', {})}")
                                    yield event
                            else:
                                if msg.content:
                                    event = {"type": "assistant", "text": msg.content, "timestamp": ts, "step": step}
                                    logger.info(f"[轨迹] step={step} assistant_len={len(msg.content)}")
                                    yield event
                    elif node == "tools":
                        msg = data["messages"][0]
                        if isinstance(msg, ToolMessage):
                            text = msg.content or ""
                            event = {
                                "type": "observation",
                                "text": text,
                                "tool_call_id": msg.tool_call_id,
                                "timestamp": ts,
                                "step": step,
                            }
                            logger.info(f"[轨迹] step={step} observation_len={len(text)} tool_call_id={msg.tool_call_id}")
                            yield event
            elapsed = time.time() - start_time
            logger.info(f"[轨迹] Agent 完成 | 总耗时={elapsed:.1f}s | 步骤={step}")
        except Exception as e:
            logger.error(f"[轨迹] Agent 执行失败: {e}", exc_info=True)
            yield {"type": "error", "text": str(e), "timestamp": round(time.time() - start_time, 2), "step": step}

    def get_history(self) -> List[Dict[str, Any]]:
        """获取当前会话的完整对话历史."""
        try:
            state = self.agent.get_state(self._get_run_config())
            messages = state.values.get("messages", [])
            return self._messages_to_dicts(messages)
        except Exception as e:
            logger.warning(f"读取历史失败: {e}")
            return []

    def clear_history(self) -> None:
        """清空对话历史（开启新会话）."""
        old_id = self.thread_id
        self.thread_id = uuid.uuid4().hex[:8]
        self.phase_ctrl.reset()
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
        """计算当前会话的上下文窗口使用率."""
        if tiktoken is None:
            return {"used_tokens": 0, "limit_tokens": 128000, "percentage": 0.0}
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {"used_tokens": 0, "limit_tokens": 128000, "percentage": 0.0}

        total_tokens = 0
        if self.system_prompt:
            total_tokens += len(encoder.encode(self.system_prompt))

        try:
            state = self.agent.get_state(self._get_run_config())
            messages = state.values.get("messages", [])
            for msg in messages:
                content = msg.content if hasattr(msg, "content") else ""
                if content:
                    total_tokens += len(encoder.encode(content))
        except Exception:
            pass

        limit = 128000
        percentage = round(total_tokens / limit * 100, 1)
        return {
            "used_tokens": total_tokens,
            "limit_tokens": limit,
            "percentage": percentage,
        }

    def get_stats(self) -> dict:
        """获取 Agent 运行统计."""
        history = self.get_history()
        return {
            "message_count": len(history),
            "tool_count": len(self.tools),
            "tools": [t.name for t in self.tools],
            "streaming": self.streaming,
            "type": "langgraph",
            "thread_id": self.thread_id,
        }
