"""上下文压缩器.

负责将消息历史列表压缩到 token 预算内，
保留最近轮次的完整上下文，对历史轮次做去重和摘要.

由 LangGraphAgent 显式调用（compact / _maybe_compact），
不再直接参与 LangGraph 节点执行.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

from ai_coding.logger import get_logger

logger = get_logger(__name__)

# 默认 token 预算 (预留 28k 给系统提示、输出和余量)
DEFAULT_TOKEN_BUDGET = 100_000

# 默认保留最近 N 轮完整交互（以 HumanMessage 为界）
DEFAULT_KEEP_RECENT_TURNS = 2

# 工具消息超过此长度时触发摘要
TOOL_SUMMARY_THRESHOLD = 2_000

# read_file 摘要保留的首尾行数
FILE_PREVIEW_LINES = 5

# grep 摘要保留的最大匹配数
GREEP_PREVIEW_MATCHES = 10


class ContextCompressor:
    """上下文压缩器.

    职责：将消息历史列表压缩到 token 预算内。
    由 LangGraphAgent 显式调用，压缩结果替换 AgentState 中的原始消息。

    压缩策略（按优先级）：
    1. 如果在预算内，原样返回
    2. 合并 older turns 中对同一文件的重复读取
    3. 对 older turns 的长 ToolMessage 做结构化摘要
    4. 如果还超预算，将 older turns 整体压缩成一条 SystemMessage 摘要
    5. 极端情况下截断（兜底）
    """

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    ) -> None:
        self.token_budget = token_budget
        self.keep_recent_turns = keep_recent_turns
        self._encoder = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    def compress(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """压缩消息列表到 token 预算内.

        Args:
            messages: 完整的消息历史列表.

        Returns:
            压缩后的消息列表，不改变原始列表.
            原始 SystemMessage（系统提示）始终保留在列表最前.
        """
        compressed, _stats = self.compress_with_stats(messages)
        return compressed

    def compress_with_stats(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], Dict[str, Any]]:
        """压缩消息列表并返回统计信息.

        Args:
            messages: 完整的消息历史列表.

        Returns:
            (压缩后的消息列表, 统计字典)
        """
        stats: Dict[str, Any] = {
            "original_count": len(messages),
            "compressed_count": len(messages),
            "original_tokens": 0,
            "compressed_tokens": 0,
            "strategies_applied": [],
        }
        if not messages:
            return list(messages), stats

        current_tokens = self._estimate_tokens(messages)
        stats["original_tokens"] = current_tokens

        if current_tokens <= self.token_budget:
            logger.debug(f"[Memory] 无需压缩 | tokens={current_tokens}")
            stats["compressed_tokens"] = current_tokens
            return list(messages), stats

        logger.info(
            f"[Memory] 开始压缩 | 原始消息={len(messages)} 条 | "
            f"tokens={current_tokens} | budget={self.token_budget}"
        )

        # 分离 SystemMessage（系统提示，始终保留）
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # 区分 recent（保留完整）和 older（可压缩）
        recent, older = self._split_turns(non_system)

        # Step 1: 去重 older 中的重复文件读取
        recent_files = set()
        for msg in recent:
            if isinstance(msg, ToolMessage):
                path = self._extract_read_file_path(msg.content)
                if path:
                    recent_files.add(path)
        older = self._deduplicate_file_reads(older, recent_files)
        tokens_after_dedup = self._estimate_tokens(system_msgs + recent + older)
        if len(older) != len([m for m in non_system if m not in recent]):
            stats["strategies_applied"].append("dedup")
        logger.info(f"[Memory] 去重后 | tokens={tokens_after_dedup}")

        # Step 2: 对 older 的长 ToolMessage 做结构化摘要
        older_before_summary = list(older)
        older = self._summarize_tool_messages(older)
        if any(
            o1.content != o2.content
            for o1, o2 in zip(older_before_summary, older)
        ):
            stats["strategies_applied"].append("summarize")
        tokens_after_summary = self._estimate_tokens(system_msgs + recent + older)
        logger.info(f"[Memory] ToolMessage 摘要后 | tokens={tokens_after_summary}")

        combined = recent + older

        # Step 3: 如果还超预算，将 older turns 整体压缩成一条 summary
        if self._estimate_tokens(system_msgs + combined) > self.token_budget:
            older_summary = self._create_turns_summary(older)
            combined = recent + [older_summary]
            stats["strategies_applied"].append("turns_summary")
            tokens_after_turn_summary = self._estimate_tokens(system_msgs + combined)
            logger.info(
                f"[Memory] 轮次摘要后 | tokens={tokens_after_turn_summary} | "
                f"recent={len(recent)} 条 + summary=1 条"
            )

        # Step 4: 极端兜底——如果 still 超预算（通常是 recent 中的 ToolMessage 太大）
        final_tokens = self._estimate_tokens(system_msgs + combined)
        if final_tokens > self.token_budget:
            sys_in_combined = [m for m in combined if isinstance(m, SystemMessage)]
            non_system_combined = [m for m in combined if not isinstance(m, SystemMessage)]
            non_system_combined = self._truncate_messages(non_system_combined)
            combined = sys_in_combined + non_system_combined
            stats["strategies_applied"].append("truncate")
            logger.warning(
                f"[Memory] 触发截断 | final_tokens={self._estimate_tokens(system_msgs + combined)}"
            )

        # 组装最终结果
        non_system_result = [m for m in combined if not isinstance(m, SystemMessage)]
        summary_in_combined = [
            m for m in combined
            if isinstance(m, SystemMessage) and m not in system_msgs
        ]
        result = system_msgs + summary_in_combined + non_system_result

        compressed_tokens = self._estimate_tokens(result)
        stats["compressed_count"] = len(result)
        stats["compressed_tokens"] = compressed_tokens

        logger.info(
            f"[Memory] 压缩完成 | {len(messages)} 条 -> {len(result)} 条 | "
            f"tokens={current_tokens} -> {compressed_tokens}"
        )
        return result, stats

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _split_turns(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[BaseMessage]]:
        """按 HumanMessage 为界划分轮次，保留最近 N 轮完整."""
        human_indices = [
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        ]
        if len(human_indices) <= self.keep_recent_turns:
            return messages, []

        split_idx = human_indices[-self.keep_recent_turns]
        return messages[split_idx:], messages[:split_idx]

    def _deduplicate_file_reads(
        self, messages: List[BaseMessage], seen_files: Optional[set] = None
    ) -> List[BaseMessage]:
        """如果 older 中有多次 read_file 同一文件，只保留最后一次.

        Args:
            messages: 待去重的消息列表（通常是 older turns）.
            seen_files: 已经在外部（如 recent turns）见过的文件路径集合.
        """
        seen_files = seen_files or set()
        result: List[BaseMessage] = []

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                path = self._extract_read_file_path(msg.content)
                if path and path in seen_files:
                    continue  # 跳过重复
                if path:
                    seen_files.add(path)
            result.append(msg)

        return list(reversed(result))

    def _summarize_tool_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """对 older turns 中的长 ToolMessage 做结构化摘要."""
        summarized: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and len(msg.content or "") > TOOL_SUMMARY_THRESHOLD:
                new_content = self._summarize_single_tool_result(msg)
                summarized.append(
                    ToolMessage(content=new_content, tool_call_id=msg.tool_call_id)
                )
            else:
                summarized.append(msg)
        return summarized

    def _summarize_single_tool_result(self, msg: ToolMessage) -> str:
        """按工具类型做结构化摘要（基于 content 启发式判断）.

        对于 read_file 结果，只保留元信息，具体内容由 file_snapshots 提供.
        """
        content = msg.content or ""
        lines = content.split("\n")
        total_lines = len(lines)

        # 1. read_file 结果 → 只保留元信息（因为 file_snapshots 会提供最新内容）
        if content.startswith("文件:"):
            path = lines[0].replace("文件:", "").strip() if lines else ""
            return f"文件: {path}\n[内容已归档至文件快照，此处省略]"

        # 2. 启发式判断：多行 + 有代码特征 → 按 read_file 摘要（兜底，处理未被识别的情况）
        if total_lines > FILE_PREVIEW_LINES * 2 and self._looks_like_code(content):
            head = "\n".join(lines[:FILE_PREVIEW_LINES])
            tail = "\n".join(lines[-FILE_PREVIEW_LINES:])
            return (
                f"[文件内容摘要: 共 {total_lines} 行]\n"
                f"--- 开头 ---\n{head}\n"
                f"...（省略 {total_lines - FILE_PREVIEW_LINES * 2} 行）...\n"
                f"--- 结尾 ---\n{tail}"
            )

        # 3. 启发式判断：grep 结果
        if total_lines > GREEP_PREVIEW_MATCHES and self._looks_like_grep(content):
            preview = "\n".join(lines[:GREEP_PREVIEW_MATCHES])
            return (
                f"[搜索结果摘要: 共 {total_lines} 条匹配]\n"
                f"{preview}\n...（省略其余 {total_lines - GREEP_PREVIEW_MATCHES} 条）..."
            )

        # 4. 兜底：长文本但非特定格式
        return content[:500] + f"\n...（省略 {len(content) - 500} 字符）..."

    def _create_turns_summary(self, messages: List[BaseMessage]) -> SystemMessage:
        """将多轮 older messages 压缩成一条 SystemMessage.

        简单规则版：提取 AIMessage 中的 tool_calls 形成行动摘要.
        """
        actions: List[str] = []
        files_accessed: set = set()

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {})
                    actions.append(f"- 调用 {name}({args})")

                    # 提取文件路径（常见参数名）
                    for key in ("path", "file_path", "target_file"):
                        if key in args and isinstance(args[key], str):
                            files_accessed.add(args[key])

        lines = ["## 此前对话摘要", "此前你已完成以下工作:"]
        if actions:
            # 去重并限制数量
            unique_actions = list(dict.fromkeys(actions))[-20:]
            lines.extend(unique_actions)
        if files_accessed:
            lines.append(f"\n已访问的文件: {', '.join(sorted(files_accessed))}")

        return SystemMessage(content="\n".join(lines))

    def _truncate_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """极端兜底：优先摘要/移除 oldest 的 ToolMessage，保留对话骨架."""
        result = list(messages)

        while result and self._estimate_tokens(result) > self.token_budget:
            removed = False

            # 策略 1: 优先移除 oldest 的 ToolMessage
            for i, m in enumerate(result):
                if isinstance(m, ToolMessage):
                    result.pop(i)
                    removed = True
                    break

            if removed:
                continue

            # 策略 2: 对 oldest 的长 AIMessage 做截断
            for i, m in enumerate(result):
                if isinstance(m, AIMessage) and m.tool_calls and len(m.content) > 200:
                    names = [tc.get("name", "") for tc in m.tool_calls]
                    result[i] = AIMessage(
                        content=f"[调用工具: {', '.join(names)}]",
                        tool_calls=m.tool_calls,
                    )
                    removed = True
                    break

            if removed:
                continue

            # 策略 3: 移除 oldest 的非 System/Human 消息
            for i, m in enumerate(result):
                if not isinstance(m, (SystemMessage, HumanMessage)):
                    result.pop(i)
                    removed = True
                    break

            if removed:
                continue

            # 策略 4: 最后手段——移除 oldest 消息
            result.pop(0)

        return result

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """估算消息列表的 token 数."""
        if not self._encoder:
            # 无 tiktoken 时按字符粗略估算
            total = 0
            for msg in messages:
                content = msg.content if hasattr(msg, "content") else ""
                total += len(content) // 4  # 粗略 4 字符 ≈ 1 token
            return total

        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else ""
            if content:
                total += len(self._encoder.encode(content))
        return total

    def _extract_read_file_path(self, content: str) -> Optional[str]:
        """从 ToolMessage content 中提取 read_file 的文件路径（启发式）."""
        # 匹配 read_file 工具的输出格式: "文件: src/main.py\n======..."
        # 或已被摘要后的格式: "文件: src/main.py\n[内容已归档...]"
        if content.startswith("文件:"):
            first_line = content.split("\n")[0]
            candidate = first_line.replace("文件:", "").strip()
            if candidate:
                return candidate
        # 备选前缀
        for prefix in ("File:", "path:", "路径:"):
            if prefix in content:
                parts = content.split(prefix, 1)
                if len(parts) == 2:
                    candidate = parts[1].split("\n")[0].strip()
                    if candidate and not candidate.startswith("-"):
                        return candidate
        return None

    def _looks_like_code(self, content: str) -> bool:
        """启发式判断内容是否像代码文件."""
        code_indicators = ["def ", "class ", "import ", "# ", "// ", "{", "}", "    "]
        lines = content.split("\n")
        score = sum(1 for line in lines for ind in code_indicators if ind in line)
        return score >= 3

    def _looks_like_grep(self, content: str) -> bool:
        """启发式判断内容是否像 grep 结果."""
        # grep 结果通常包含行号和冒号，如 "src/main.py:42:def foo():"
        lines = content.split("\n")
        match_pattern_count = sum(
            1 for line in lines if ":" in line and len(line.split(":")) >= 2
        )
        return match_pattern_count >= 3
