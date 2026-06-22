"""记忆提取器.

从对话消息中提取结构化长期记忆.
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ai_coding.logger import get_logger
from ai_coding.memory.models import MemoryEntry, MemoryScope, MemoryType

logger = get_logger(__name__)


_EXTRACTION_SYSTEM_PROMPT = """你是一名记忆提取助手。请从以下对话中识别出应该长期保留的用户偏好、项目规则、关键事实或架构决策。

只提取满足以下条件的信息：
1. 用户明确表达为长期有效，例如使用了“一直”“以后”“默认”“总是”“不要”“必须”等词。
2. 对后续会话有实际帮助，例如回复语言、代码风格、项目结构、技术选型、命名约定等。
3. 不是一次性请求或临时指令（如“这次用英文回复”）。

作用域（scope）选择规则：
- "user"：跨项目生效的用户偏好，例如回复语言、代码风格、解释详细程度、命名习惯等。
- "project"：与当前项目强相关的规则/事实/架构，例如技术栈、项目结构、构建命令等。

输出格式为 JSON：
{
  "memories": [
    {
      "content": "记忆内容，简洁一句话",
      "type": "preference|rule|fact|architecture",
      "scope": "user|project",
      "confidence": 0.9,
      "explicit": true
    }
  ]
}

如果没有值得长期记忆的内容，返回 {"memories": []}。
"""


class MemoryExtractor:
    """从会话对话中提取长期记忆.

    调用 LLM 识别用户显式偏好、项目规则与关键事实，输出 MemoryEntry 列表.
    """

    def __init__(
        self,
        llm_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._llm: Optional[Any] = None

    def _get_llm(self) -> Optional[Any]:
        """延迟创建 LLM 实例."""
        if self._llm is None and self._llm_factory is not None:
            self._llm = self._llm_factory()
        return self._llm

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """将消息列表格式化为纯文本供提取使用."""
        lines: List[str] = []
        for msg in messages:
            role = "unknown"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, BaseMessage):
                # AIMessage / ToolMessage 都继承 BaseMessage
                msg_type = msg.type if hasattr(msg, "type") else ""
                if msg_type == "ai":
                    role = "assistant"
                elif msg_type == "tool":
                    role = "tool"
                else:
                    role = "assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON，提取 memories 列表."""
        text = content.strip()
        # 如果 LLM 用 markdown code block 包裹，尝试提取
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"记忆提取结果 JSON 解析失败: {e}\n内容: {content[:200]}")
            return []

        if not isinstance(data, dict):
            logger.warning(f"记忆提取结果不是 JSON 对象: {content[:200]}")
            return []

        memories = data.get("memories", [])
        if not isinstance(memories, list):
            logger.warning(f"记忆提取结果中 memories 字段不是列表: {content[:200]}")
            return []

        return memories

    def _build_entries(
        self,
        raw_memories: List[Dict[str, Any]],
        session_id: str,
    ) -> List[MemoryEntry]:
        """将原始记忆字典转换为 MemoryEntry 列表."""
        entries: List[MemoryEntry] = []
        now = time.time()
        for raw in raw_memories:
            content = str(raw.get("content", "")).strip()
            if not content:
                continue

            try:
                mem_type = MemoryType(str(raw.get("type", MemoryType.FACT.value)))
            except ValueError:
                mem_type = MemoryType.FACT

            try:
                scope = MemoryScope(str(raw.get("scope", MemoryScope.PROJECT.value)))
            except ValueError:
                scope = MemoryScope.PROJECT

            confidence = float(raw.get("confidence", 0.5))
            explicit = bool(raw.get("explicit", False))

            # 第一版只保留显式声明或高置信度的记忆
            if not explicit and confidence < 0.7:
                logger.debug(f"丢弃低置信度非显式记忆: {content}")
                continue

            entries.append(
                MemoryEntry(
                    content=content,
                    type=mem_type,
                    scope=scope,
                    confidence=min(max(confidence, 0.0), 1.0),
                    observation_count=1,
                    explicit=explicit,
                    source_sessions=[session_id] if session_id else [],
                    created_at=now,
                    updated_at=now,
                )
            )
        return entries

    def extract(
        self,
        messages: List[BaseMessage],
        session_id: str = "",
    ) -> List[MemoryEntry]:
        """从消息列表中提取记忆.

        Args:
            messages: 对话消息列表，元素通常为 langchain_core.messages.BaseMessage.
            session_id: 来源会话 ID.

        Returns:
            提取出的 MemoryEntry 列表.
        """
        if not messages:
            return []

        llm = self._get_llm()
        if llm is None:
            logger.warning("MemoryExtractor 未配置 LLM，跳过记忆提取")
            return []

        conversation_text = self._format_messages(messages)
        system_msg = SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT)
        human_msg = HumanMessage(
            content=f"请从以下对话中提取长期记忆：\n\n{conversation_text}"
        )

        try:
            response = llm.invoke([system_msg, human_msg])
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            logger.warning(f"调用 LLM 提取记忆失败: {e}")
            return []

        raw_memories = self._parse_response(content)
        if not raw_memories:
            logger.debug("本次对话未提取到长期记忆")
            return []

        entries = self._build_entries(raw_memories, session_id)
        logger.info(f"从会话 [{session_id}] 提取到 {len(entries)} 条记忆")
        return entries
