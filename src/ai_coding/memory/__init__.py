"""长期记忆模块.

提供 Agent 跨会话记忆能力：
- MemoryEntry: 结构化记忆条目
- MemoryStore: 本地持久化存储
- MemoryExtractor: 从对话中提取记忆
- MemoryRetriever: 检索并格式化记忆摘要
"""

from ai_coding.memory.extractor import MemoryExtractor
from ai_coding.memory.models import MemoryEntry, MemoryScope, MemoryType
from ai_coding.memory.retriever import MemoryRetriever
from ai_coding.memory.store import MemoryStore

__all__ = [
    "MemoryEntry",
    "MemoryScope",
    "MemoryType",
    "MemoryStore",
    "MemoryExtractor",
    "MemoryRetriever",
]
