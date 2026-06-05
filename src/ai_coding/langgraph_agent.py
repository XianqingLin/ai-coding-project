"""LangGraph ReAct Agent 实现.

手动构建 StateGraph，实现推理-行动循环.
使用 MemorySaver 实现对话记忆持久化.
支持 reasoning_content 提取（通过 KimiChatOpenAI 子类）.
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

    # 探索类工具名称（用于统计探索步数和软约束提醒）
    EXPLORE_TOOL_NAMES = {"read_file", "list_dir", "grep"}

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

        # Agent 阶段状态（由 plan 工具触发进入修改阶段）
        self.has_plan: bool = False
        self.plan_content: str = ""

        # 初始化工具注册表（用于自定义 tools_node）
        from ai_coding.tools.base import ToolRegistry
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            self.tool_registry.register(tool)

        # 构建 LangGraph（绑定 MemorySaver）
        self.agent = self._build_graph()

        logger.info(
            f"LangGraph Agent 初始化完成 | 工具: {[t.name for t in self.tools]} | "
            f"流式: {streaming} | 会话: {self.thread_id}"
        )

    def _count_explore_steps(self, messages: List[BaseMessage]) -> int:
        """统计历史消息中探索类工具（read_file/list_dir/grep）的调用次数.

        仅用于软约束提醒，不再用于强制阶段切换.

        """
        count = 0
        for m in messages:
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            names = {tc.get("name", "") for tc in m.tool_calls}
            if names & self.EXPLORE_TOOL_NAMES:
                count += 1
        return count

    def _build_explore_hint(self, explore_steps: int) -> str:
        """根据探索步数生成软约束提醒文本.

        在探索工具结果后追加提示，引导 Agent 尽快调用 plan 进入修改阶段.
        措辞从建议逐步升级为要求，降低 LLM 的\"自主选择\"偏差.

        """
        if self.has_plan:
            return ""
        if explore_steps <= 2:
            return ""
        if explore_steps <= 5:
            return (
                f"[提示] 这是你第 {explore_steps} 次探索。"
                f"在读了 3-5 个关键文件后，你就应该已经足够了解代码。"
                f"现在必须调用 plan 工具提交修改计划，然后开始修改。"
            )
        if explore_steps <= 10:
            return (
                f"[要求] 你已探索 {explore_steps} 步，步数已过多。"
                f"无论你认为自己是否已完全理解代码，现在都必须调用 plan 工具。"
                f"plan 不需要完美，你可以在 str_replace_file 执行过程中调整。"
            )
        return (
            f"[强制] 你已探索 {explore_steps} 步，严重超时。"
            f"这是系统要求：你的下一步必须是调用 plan 工具提交修改计划。"
            f"不调用 plan 就无法开始修改，而你没有更多步数可以浪费在探索上。"
        )

    def _build_graph(self):
        """构建 ReAct 图结构（支持 plan 工具触发的自主阶段切换）."""
        # 转换所有工具为 LangChain 格式
        all_lc_tools = [t.to_langchain_tool() for t in self.tools]
        bound_llm = self.llm.bind_tools(all_lc_tools)

        # 自定义工具节点：处理 plan 工具，对探索工具追加软约束提醒
        def tools_node(state: MessagesState):
            """执行工具调用，处理 plan 状态，追加软约束提醒."""
            last_msg = state["messages"][-1]
            if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                return {"messages": []}

            tool_messages = []
            explore_steps = self._count_explore_steps(state["messages"])

            for tc in last_msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                tool_id = tc.get("id", "")

                # 确保 args 是 dict
                if hasattr(args, "dict"):
                    args = args.dict()
                elif not isinstance(args, dict):
                    args = dict(args)

                # 处理 plan 工具：标记进入修改阶段
                if name == "plan":
                    self.has_plan = True
                    self.plan_content = args.get("plan", "")
                    logger.info(f"[Plan] Agent 提交修改计划: {self.plan_content[:200]}")

                result = self.tool_registry.execute(name, args)

                # 软约束：探索工具的结果后追加提醒
                if name in self.EXPLORE_TOOL_NAMES:
                    if not self.has_plan:
                        hint = self._build_explore_hint(explore_steps + 1)
                        if hint:
                            result += f"\n\n{hint}"
                    else:
                        # 已提交 plan，但仍使用探索工具 → 催促立即 edit
                        result += (
                            "\n\n[提醒] 你已提交修改计划，当前处于修改阶段。"
                            "你的下一步必须是使用 str_replace_file 或 write_file 执行修改，"
                            "而不是继续探索。请立即开始修改代码。"
                        )

                tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

            return {"messages": tool_messages}

        def agent_node(state: MessagesState):
            """Agent 节点：调用 LLM，返回 AI 消息."""
            messages = list(state["messages"])

            # 计算当前步数（已完成的 AI 决策轮数）
            current_step = len([m for m in messages if isinstance(m, AIMessage)])

            # 在消息开头添加系统提示（只在首轮添加）
            if self.system_prompt and not any(
                isinstance(m, (AIMessage, ToolMessage)) for m in messages
            ):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            # 滑动窗口截断：保留最近 N 轮对话，防止消息无限累积
            messages = self._apply_sliding_window(messages, max_turns=8)

            # 当消息累积较大时自动压缩，防止上下文膨胀导致 LLM 延迟激增
            total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
            if total_chars > 40000:
                messages = self._compress_messages(messages)

            # 阶段标签（仅用于日志，不影响工具可用性）
            explore_steps = self._count_explore_steps(messages)
            if self.has_plan:
                phase_tag = f"EDIT(plan) explore={explore_steps}"
            else:
                phase_tag = f"EXPLORE({explore_steps})"

            # 记录 LLM 输入摘要
            last_msg = messages[-1] if messages else None
            last_content = last_msg.content[:200] if last_msg and hasattr(last_msg, "content") else ""
            logger.info(
                f"[LLM IN]  messages={len(messages)} step={current_step + 1} "
                f"phase={phase_tag} last_content={last_content[:100]!r}"
            )
            t0 = time.time()

            # 调用 LLM（所有阶段统一使用完整工具集）
            response = bound_llm.invoke(messages)

            elapsed = time.time() - t0
            # 记录 LLM 输出摘要
            if isinstance(response, AIMessage):
                tool_names = [tc.get("name", "") for tc in response.tool_calls]
                reasoning = response.additional_kwargs.get("reasoning_content", "")
                logger.info(
                    f"[LLM OUT] elapsed={elapsed:.2f}s "
                    f"content_len={len(response.content)} "
                    f"tool_calls={tool_names} "
                    f"reasoning_len={len(reasoning) if reasoning else 0}"
                )
                logger.debug(f"[LLM OUT] content={response.content[:500]!r}")

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
        builder.add_node("tools", tools_node)
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
        # recursion_limit 需要足够大：
        # 每轮 ReAct = agent(AIMessage) + tools(ToolMessage) = 2 个事件
        # 再加上可能的阶段切换消息，预留 5 倍余量
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.max_iterations * 5 + 20,
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
        - {"type": "thinking", "text": "...", "timestamp": ...}      - AI 的思考过程
        - {"type": "assistant", "text": "...", "timestamp": ...}     - AI 的最终回复
        - {"type": "tool_call", "name": "...", "args": {...}, "timestamp": ...} - 工具调用
        - {"type": "observation", "text": "...", "timestamp": ...}   - 工具执行结果

        Args:
            user_input: 用户的自然语言输入.

        Yields:
            工作流事件字典（包含 timestamp 字段）.

        """
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
                                # 有工具调用时，优先从 reasoning_content 提取思考过程（Kimi K2.6）
                                thinking_text = msg.additional_kwargs.get("reasoning_content") or msg.content
                                if thinking_text:
                                    event = {"type": "thinking", "text": thinking_text, "timestamp": ts, "step": step}
                                    logger.info(f"[轨迹] step={step} thinking_len={len(thinking_text)}")
                                    yield event
                                # 输出工具调用
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
                                # 无工具调用时，content 是最终回复
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

    def _apply_sliding_window(
        self, messages: List[BaseMessage], max_turns: int = 8
    ) -> List[BaseMessage]:
        """滑动窗口截断：保留最近 N 轮完整 agent-tool 对话，更早的折叠为摘要.

        LangGraph 的 MessagesState 会永久累积所有历史消息，导致 LLM 输入
        越来越长、响应越来越慢。此方法在调用 LLM 前截断消息列表。

        关键约束：一个 AIMessage 可能触发多个 ToolMessage（如同时调用
        多个 read_file），截断时必须保留完整的 "AIMessage + 对应的所有
        ToolMessage"，否则会出现 tool_call_id 不匹配导致 API 400 错误。

        Checkpoint（MemorySaver）中的完整历史不受影响，仅影响传给 LLM
        的上下文窗口。

        Args:
            messages: 原始消息列表.
            max_turns: 保留的最近对话轮数，默认 8.

        Returns:
            截断后的消息列表.
        """
        preserved: List[BaseMessage] = []
        conversation: List[BaseMessage] = []

        for m in messages:
            if isinstance(m, (SystemMessage, HumanMessage)):
                preserved.append(m)
            else:
                conversation.append(m)

        # 从后往前数完整的轮次
        # 一轮 = 1 个 AIMessage + 它对应的所有 ToolMessage
        kept: List[BaseMessage] = []
        turns = 0
        i = len(conversation) - 1

        while i >= 0 and turns < max_turns:
            # 1. 从后往前收集所有连续的 ToolMessage（属于当前轮）
            while i >= 0 and isinstance(conversation[i], ToolMessage):
                kept.insert(0, conversation[i])
                i -= 1

            # 2. 收集对应的 AIMessage
            if i >= 0 and isinstance(conversation[i], AIMessage):
                kept.insert(0, conversation[i])
                i -= 1
                turns += 1
            else:
                # 结构异常（可能是最终答案的 AIMessage，后面没有 ToolMessage）
                # 直接保留并停止回溯
                if i >= 0:
                    kept.insert(0, conversation[i])
                    i -= 1
                break

        if i >= 0:
            dropped = conversation[: i + 1]
            dropped_tool_calls = sum(
                1 for m in dropped if isinstance(m, AIMessage) and m.tool_calls
            )
            dropped_obs = sum(1 for m in dropped if isinstance(m, ToolMessage))

            summary = SystemMessage(
                content=(
                    f"[历史摘要] 前 {len(dropped)} 条消息已折叠 "
                    f"({dropped_tool_calls} 次工具调用, {dropped_obs} 次观察结果). "
                    f"保留最近 {turns} 轮完整上下文."
                )
            )
            result = preserved + [summary] + kept
            logger.info(
                f"[SLIDE] messages={len(messages)} -> {len(result)} "
                f"(保留最近 {turns} 轮, 折叠 {len(dropped)} 条)"
            )
            return result

        return messages

    def _compress_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """压缩历史消息中的大观察结果，防止上下文膨胀.

        将长度超过 3000 字符的 ToolMessage 截断为"前 600 + 后 400"的摘要形式，
        保留文件名和关键信息的同时大幅降低上下文大小.
        """
        compressed = []
        total_before = 0
        total_after = 0
        for m in messages:
            content = m.content if hasattr(m, "content") else ""
            total_before += len(content)
            if isinstance(m, ToolMessage) and len(content) > 3000:
                prefix = content[:600]
                suffix = content[-400:]
                summary = (
                    f"{prefix}\n"
                    f"\n... [内容已压缩，原长度 {len(content)} 字符] ...\n"
                    f"{suffix}"
                )
                compressed.append(ToolMessage(content=summary, tool_call_id=m.tool_call_id))
                total_after += len(summary)
            else:
                compressed.append(m)
                total_after += len(content)
        logger.info(
            f"[COMPRESS] before={total_before} after={total_after} "
            f"saved={total_before - total_after} ({(total_before - total_after) / max(total_before, 1) * 100:.1f}%)"
        )
        return compressed

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

        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "default_system_prompt.txt"
        template = prompt_path.read_text(encoding="utf-8")

        tool_descriptions = []
        for tool in self.tools:
            params = ", ".join(p.name for p in tool.parameters)
            tool_descriptions.append(f"  - {tool.name}({params}): {tool.description}")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "  (暂无可用工具)"

        return template.format(tools_text=tools_text)

    def get_context_usage(self) -> dict:
        """计算当前会话的上下文窗口使用率.

        使用 tiktoken (cl100k_base) 估算已用 token 数.

        Returns:
            包含 used_tokens, limit_tokens, percentage 的字典.

        """
        if tiktoken is None:
            return {"used_tokens": 0, "limit_tokens": 128000, "percentage": 0.0}
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
