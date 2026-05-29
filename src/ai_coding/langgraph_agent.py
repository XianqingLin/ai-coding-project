"""LangGraph ReAct Agent 实现.

手动构建 StateGraph，实现推理-行动循环.
在 agent 节点中自动注入 reasoning_content，兼容 Kimi K2.6.
使用 MemorySaver 实现对话记忆持久化.
"""

import uuid
from typing import Any, Dict, Iterator, List, Optional

import tiktoken

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode

from ai_coding.logger import get_logger
from ai_coding.tools.base import Tool

logger = get_logger(__name__)


class LangGraphAgent:
    """基于 LangGraph 的 ReAct Agent（带记忆）.

    手动构建图结构：
      START -> agent -> [有 tool_calls ?] -> tools -> agent -> ...
                          [无 tool_calls]  -> END

    使用 MemorySaver checkpoint 机制实现跨调用的对话记忆.
    每个 thread_id 对应一个独立的会话上下文.

    Attributes:
        llm: LangChain ChatOpenAI 实例.
        tools: 自定义工具列表.
        agent: 编译后的 LangGraph.
        max_iterations: 最大迭代次数.
        streaming: 是否启用流式输出.
        thread_id: 当前会话标识.

    """

    def __init__(
        self,
        llm,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        streaming: bool = True,
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """初始化 LangGraph Agent.

        Args:
            llm: LangChain ChatOpenAI 实例.
            tools: 自定义工具列表.
            max_iterations: 最大 ReAct 迭代次数.
            streaming: 是否启用流式输出.
            system_prompt: 自定义系统提示. 为 None 时使用默认提示.
            thread_id: 会话标识. 为 None 时自动生成.

        """
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.system_prompt = system_prompt or self._build_system_prompt()
        self.thread_id = thread_id or uuid.uuid4().hex[:8]

        # 构建 LangGraph（绑定 MemorySaver）
        self.agent = self._build_graph()

        logger.info(
            f"LangGraph Agent 初始化完成 | 工具: {[t.name for t in self.tools]} | "
            f"流式: {streaming} | 会话: {self.thread_id}"
        )

    def _build_graph(self):
        """构建 ReAct 图结构."""
        # 转换工具为 LangChain 格式
        lc_tools = [t.to_langchain_tool() for t in self.tools]
        tool_node = ToolNode(lc_tools)

        # 绑定工具到 LLM
        llm_with_tools = self.llm.bind_tools(lc_tools)

        def agent_node(state: MessagesState):
            """Agent 节点：调用 LLM，返回 AI 消息."""
            messages = list(state["messages"])

            # 在消息开头添加系统提示（只在首轮添加）
            if self.system_prompt and not any(
                isinstance(m, (AIMessage, ToolMessage)) for m in messages
            ):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            # 调用 LLM
            response = llm_with_tools.invoke(messages)

            # 修复：为包含 tool_calls 的 AI 消息注入空的 reasoning_content
            # 这是 Kimi K2.6 的要求，否则后续请求会报 400 错误
            if isinstance(response, AIMessage) and response.tool_calls:
                response.additional_kwargs["reasoning_content"] = ""
                logger.debug(f"注入 reasoning_content 到 {response.tool_calls}")

            return {"messages": [response]}

        def should_continue(state: MessagesState):
            """条件判断：是否需要调用工具."""
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            return END

        # 构建图
        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "agent")

        # 绑定 MemorySaver checkpoint 实现记忆
        return builder.compile(checkpointer=MemorySaver())

    def _get_run_config(self) -> dict:
        """构建运行配置（包含 thread_id 和 recursion_limit）."""
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.max_iterations * 2 + 5,
        }

    def _build_input(self, user_input: str) -> dict:
        """构建 Agent 输入消息."""
        return {"messages": [HumanMessage(content=user_input)]}

    def run(self, user_input: str) -> str:
        """处理用户输入 (非流式).

        Args:
            user_input: 用户的自然语言输入.

        Returns:
            Agent 的最终回复.

        """
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
        """处理用户输入 (流式输出).

        使用 LangGraph 的 stream 模式实时返回 AI 生成的内容.

        Args:
            user_input: 用户的自然语言输入.

        Yields:
            每个文本片段 (delta content).

        """
        logger.info(f"[流式] 用户输入: {user_input[:100]}")

        try:
            for event in self.agent.stream(
                self._build_input(user_input),
                config=self._get_run_config(),
                stream_mode="messages",
            ):
                msg, metadata = event

                # 只输出 agent 节点生成的内容
                if metadata.get("langgraph_node") == "agent":
                    content = msg.content if hasattr(msg, "content") else ""
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"[流式] Agent 执行失败: {e}", exc_info=True)
            yield f"[错误] Agent 执行失败: {e}"

    def run_with_trace(self, user_input: str) -> Iterator[Dict[str, Any]]:
        """处理用户输入，返回完整的工作流程轨迹（用于 verbose 显示）.

        使用 stream_mode="updates" 捕获每个节点的输出，yield 结构化事件：
        - {"type": "thinking", "text": "..."}      - AI 的思考过程
        - {"type": "assistant", "text": "..."}     - AI 的最终回复
        - {"type": "tool_call", "name": "...", "args": {...}} - 工具调用
        - {"type": "observation", "text": "..."}   - 工具执行结果

        Args:
            user_input: 用户的自然语言输入.

        Yields:
            工作流事件字典.

        """
        logger.info(f"[轨迹] 用户输入: {user_input[:100]}")

        try:
            for chunk in self.agent.stream(
                self._build_input(user_input),
                config=self._get_run_config(),
                stream_mode="updates",
            ):
                for node, data in chunk.items():
                    if node == "agent":
                        msg = data["messages"][0]
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                # 有工具调用时，content 是思考过程
                                if msg.content:
                                    yield {"type": "thinking", "text": msg.content}
                                # 输出工具调用
                                for tc in msg.tool_calls:
                                    yield {
                                        "type": "tool_call",
                                        "name": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                        "id": tc.get("id", ""),
                                    }
                            else:
                                # 无工具调用时，content 是最终回复
                                if msg.content:
                                    yield {"type": "assistant", "text": msg.content}

                    elif node == "tools":
                        msg = data["messages"][0]
                        if isinstance(msg, ToolMessage):
                            yield {
                                "type": "observation",
                                "text": msg.content or "",
                                "tool_call_id": msg.tool_call_id,
                            }

            logger.info("[轨迹] Agent 完成")

        except Exception as e:
            logger.error(f"[轨迹] Agent 执行失败: {e}", exc_info=True)
            yield {"type": "error", "text": str(e)}

    def get_history(self) -> List[Dict[str, Any]]:
        """获取当前会话的完整对话历史.

        从 MemorySaver checkpoint 中读取消息，转换为字典列表格式.

        Returns:
            消息字典列表，每条包含 role 和 content.

        """
        try:
            state = self.agent.get_state(self._get_run_config())
            messages = state.values.get("messages", [])
            return self._messages_to_dicts(messages)
        except Exception as e:
            logger.warning(f"读取历史失败: {e}")
            return []

    def clear_history(self) -> None:
        """清空对话历史（开启新会话）.

        生成新的 thread_id，使后续对话进入全新的会话上下文.

        """
        old_id = self.thread_id
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
        """构建默认系统提示."""
        tool_descriptions = []
        for tool in self.tools:
            params = ", ".join(p.name for p in tool.parameters)
            tool_descriptions.append(f"  - {tool.name}({params}): {tool.description}")

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "  (暂无可用工具)"

        return (
            "你是一个 AI 编程助手, 专门帮助用户进行代码开发任务.\n"
            "你可以使用以下工具来完成任务:\n"
            f"{tools_text}\n\n"
            "工作原则:\n"
            "1. 如果任务需要查看或操作文件, 请先使用相应工具.\n"
            "2. 如果任务可以通过直接回答完成, 请不要调用工具.\n"
            "3. 每次回复尽量简洁、准确.\n"
            "4. 执行命令时请注意安全性."
        )

    def get_context_usage(self) -> dict:
        """计算当前会话的上下文窗口使用率.

        使用 tiktoken (cl100k_base) 估算已用 token 数.

        Returns:
            包含 used_tokens, limit_tokens, percentage 的字典.

        """
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {"used_tokens": 0, "limit_tokens": 128000, "percentage": 0.0}

        total_tokens = 0

        # 系统提示占用的 token
        if self.system_prompt:
            total_tokens += len(encoder.encode(self.system_prompt))

        # 历史消息占用的 token
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
