"""记忆检索器.

从 MemoryStore 中检索相关记忆并格式化为系统提示可用的摘要文本.
"""

import time
from typing import Dict, List, Optional

from ai_coding.memory.models import MemoryEntry, MemoryScope
from ai_coding.memory.store import MemoryStore


class MemoryRetriever:
    """记忆检索与格式化.

    从项目级记忆存储中读取条目，按置信度与新鲜度排序，
    格式化为 `# 记忆摘要` 段落文本.
    """

    DEFAULT_TOP_K = 10
    CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.store = store
        self.top_k = top_k

    @staticmethod
    def _effective_confidence(entry: MemoryEntry) -> float:
        """计算有效置信度，加入时间衰减.

        每 30 天衰减约 5%，避免陈旧记忆长期占据高排名.
        """
        days_since_update = (time.time() - entry.updated_at) / 86400.0
        decay = float(0.999**days_since_update)
        return float(entry.confidence * decay)

    def retrieve(
        self,
        query: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
    ) -> List[MemoryEntry]:
        """检索相关记忆.

        Args:
            query: 可选查询文本，当前版本仅用于占位，后续可做语义过滤.
            scope: 可选作用域过滤.

        Returns:
            按有效置信度降序排列的记忆列表.
        """
        entries = self.store.list(scope=scope, include_invalidated=False)
        filtered = [
            e
            for e in entries
            if self._effective_confidence(e) >= self.CONFIDENCE_THRESHOLD
        ]
        filtered.sort(key=self._effective_confidence, reverse=True)
        return filtered[: self.top_k]

    def get_summary(
        self,
        query: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
    ) -> str:
        """格式化为记忆摘要文本.

        若没有任何记忆，返回空字符串.
        """
        entries = self.retrieve(query=query, scope=scope)
        if not entries:
            return ""
        lines = ["- " + e.content for e in entries]
        return "\n".join(lines)

    _SCOPE_HEADERS = {
        MemoryScope.USER: "# 用户偏好",
        MemoryScope.PROJECT: "# 项目记忆",
    }

    @classmethod
    def combine_summaries(
        cls,
        scope_stores: Dict[MemoryScope, MemoryStore],
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """从多个作用域的存储中检索并合并为一份摘要文本."""
        parts: List[str] = []
        for scope, store in scope_stores.items():
            retriever = cls(store, top_k=top_k)
            summary = retriever.get_summary(scope=scope)
            if summary:
                header = cls._SCOPE_HEADERS.get(scope, f"# {scope.value}")
                parts.append(f"{header}\n{summary}")
        return "\n\n".join(parts)
