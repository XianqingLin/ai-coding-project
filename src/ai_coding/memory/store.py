"""记忆本地持久化存储.

按项目(work_dir)隔离记忆，存储格式为 JSONL.
路径: data/memory/projects/<work_dir_key>/memory.jsonl
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Optional

from ai_coding.logger import get_logger
from ai_coding.memory.models import MemoryEntry, MemoryScope
from ai_coding.persistence.config import get_storage_root
from ai_coding.persistence.storage import _work_dir_key

logger = get_logger(__name__)


def _default_storage_root() -> Path:
    """获取默认存储根目录."""
    return get_storage_root()


def _normalize_text(text: str) -> str:
    """归一化文本用于相似度比较."""
    return " ".join(text.lower().split())


def _tokenize_for_similarity(text: str) -> set[str]:
    """为相似度比较生成 token 集合.

    对有空格的文本使用词级 token，对无空格文本（如中文）使用字符级 token，
    保证跨语言都能获得有效重叠.
    """
    normalized = _normalize_text(text)
    if " " in normalized:
        return set(normalized.split())
    return set(normalized)


def _content_similarity(left: str, right: str) -> float:
    """计算两段文本的相似度，返回 0.0~1.0.

    策略：
    1. 完全一致 -> 1.0
    2. 互相子串 -> 0.9
    3. 否则使用 Jaccard token 重叠
    """
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.9

    left_tokens = _tokenize_for_similarity(left)
    right_tokens = _tokenize_for_similarity(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


class MemoryStore:
    """记忆存储.

    负责项目级/用户级记忆的增删改查、相似合并与持久化.
    """

    SIMILARITY_THRESHOLD = 0.7

    def __init__(
        self,
        work_dir: str = "",
        root: Optional[Path] = None,
        scope: MemoryScope = MemoryScope.PROJECT,
    ) -> None:
        self.scope = scope
        self.root = Path(root) if root else _default_storage_root()

        if self.scope == MemoryScope.USER:
            # 用户级偏好跨项目共享，使用全局存储路径
            self.work_dir = ""
            self.work_dir_key = ""
            self._memory_dir = self.root / "memory" / "global"
        else:
            # 项目级/会话级记忆按 work_dir 隔离
            if not work_dir:
                raise ValueError("PROJECT/SESSION scope 必须提供 work_dir")
            # 解析为规范路径，避免同一目录因短路径/长路径字符串不同导致 key 不一致
            self.work_dir = str(Path(work_dir).expanduser().resolve())
            self.work_dir_key = _work_dir_key(self.work_dir)
            self._memory_dir = self.root / "memory" / "projects" / self.work_dir_key

        self._memory_path = self._memory_dir / "memory.jsonl"
        self._lock = threading.RLock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """确保存储目录存在."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def _load_entries(self) -> List[MemoryEntry]:
        """加载所有记忆条目."""
        if not self._memory_path.exists():
            return []
        entries: List[MemoryEntry] = []
        try:
            with self._lock, self._memory_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(MemoryEntry.from_dict(json.loads(line)))
                    except Exception as e:
                        logger.warning(f"解析记忆条目失败: {e}")
        except Exception as e:
            logger.warning(f"加载记忆文件失败: {e}")
        return entries

    def _save_entries(self, entries: List[MemoryEntry]) -> None:
        """保存所有记忆条目."""
        try:
            self._ensure_dir()
            with self._lock, self._memory_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"保存记忆文件失败: {e}")

    def add_or_update(self, entry: MemoryEntry) -> MemoryEntry:
        """添加新记忆或合并更新已有相似记忆.

        若找到相似条目，则合并 source_sessions、提升 observation_count、
        更新 content 和 updated_at；否则追加新条目.
        """
        entries = self._load_entries()
        similar = self._find_similar(entries, entry)
        now = time.time()
        if similar is not None:
            similar.content = entry.content
            similar.confidence = max(similar.confidence, entry.confidence)
            similar.observation_count += entry.observation_count
            similar.updated_at = now
            similar.source_sessions = list(
                set(similar.source_sessions) | set(entry.source_sessions)
            )
            similar.invalidated = False
            self._save_entries(entries)
            return similar

        entry.created_at = now
        entry.updated_at = now
        entries.append(entry)
        self._save_entries(entries)
        return entry

    def list(
        self,
        scope: Optional[MemoryScope] = None,
        include_invalidated: bool = False,
    ) -> List[MemoryEntry]:
        """列出记忆条目.

        Args:
            scope: 若指定则只返回该作用域的记忆.
            include_invalidated: 是否包含已被否决的记忆.
        """
        entries = self._load_entries()
        if scope is not None:
            entries = [e for e in entries if e.scope == scope]
        if not include_invalidated:
            entries = [e for e in entries if not e.invalidated]
        return entries

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """根据 ID 获取记忆条目."""
        for entry in self._load_entries():
            if entry.id == entry_id:
                return entry
        return None

    def delete(self, entry_id: str) -> bool:
        """删除记忆条目."""
        entries = self._load_entries()
        new_entries = [e for e in entries if e.id != entry_id]
        if len(new_entries) == len(entries):
            return False
        self._save_entries(new_entries)
        return True

    def invalidate(self, entry_id: str) -> bool:
        """将记忆条目标记为已否决/失效."""
        entries = self._load_entries()
        for entry in entries:
            if entry.id == entry_id:
                entry.invalidated = True
                entry.updated_at = time.time()
                self._save_entries(entries)
                return True
        return False

    def _find_similar(
        self,
        entries: List[MemoryEntry],
        new_entry: MemoryEntry,
    ) -> Optional[MemoryEntry]:
        """在已有条目中查找与新条目最相似且未失效的条目."""
        best: Optional[MemoryEntry] = None
        best_score = 0.0
        for entry in entries:
            if entry.invalidated:
                continue
            score = _content_similarity(entry.content, new_entry.content)
            if score > best_score and score >= self.SIMILARITY_THRESHOLD:
                best = entry
                best_score = score
        return best
