"""持久化存储引擎.

管理 ~/.ai-coding/ 目录下的会话存储：
- session_index.jsonl
- sessions/<workDirKey>/<sessionId>/
  - meta.json
  - state.json
  - agents/main/wire.jsonl
  - agents/<subagentId>/wire.jsonl
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_coding.logger import get_logger
from ai_coding.persistence.config import get_storage_root

logger = get_logger(__name__)


def _default_storage_root() -> Path:
    """获取默认存储根目录."""
    return get_storage_root()


def _work_dir_key(work_dir: str) -> str:
    """将工作目录路径编码为目录安全键.

    使用 SHA256 取前 16 位十六进制，避免路径中的特殊字符.
    """
    return hashlib.sha256(work_dir.encode("utf-8")).hexdigest()[:16]


class StorageEngine:
    """会话持久化存储引擎."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _default_storage_root()
        self.index_path = self.root / "session_index.jsonl"
        # 全局可重入锁，串行化所有持久化操作，避免并发写损坏 JSON/JSONL 文件
        self._lock = threading.RLock()
        self._ensure_root()

    def _ensure_root(self) -> None:
        """确保存储根目录存在."""
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 索引管理
    # ------------------------------------------------------------------ #

    def _load_index(self) -> List[Dict[str, Any]]:
        """加载会话索引."""
        if not self.index_path.exists():
            return []
        entries = []
        try:
            with self._lock, self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"加载会话索引失败: {e}")
        return entries

    def _save_index(self, entries: List[Dict[str, Any]]) -> None:
        """保存会话索引."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.index_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"保存会话索引失败: {e}")

    def _find_index_entry(self, session_id: str) -> Optional[Dict[str, Any]]:
        """按 session_id 查找索引条目."""
        for entry in self._load_index():
            if entry.get("session_id") == session_id:
                return entry
        return None

    def upsert_index(
        self,
        session_id: str,
        work_dir: str,
        name: str,
        created_at: float,
        updated_at: Optional[float] = None,
    ) -> None:
        """插入或更新索引条目."""
        entries = self._load_index()
        found = False
        for entry in entries:
            if entry.get("session_id") == session_id:
                entry["work_dir"] = work_dir
                entry["work_dir_key"] = _work_dir_key(work_dir)
                entry["name"] = name
                entry["created_at"] = created_at
                entry["updated_at"] = updated_at or time.time()
                found = True
                break
        if not found:
            entries.append({
                "session_id": session_id,
                "work_dir": work_dir,
                "work_dir_key": _work_dir_key(work_dir),
                "name": name,
                "created_at": created_at,
                "updated_at": updated_at or time.time(),
            })
        self._save_index(entries)

    def remove_index(self, session_id: str) -> None:
        """从索引中移除条目."""
        entries = [e for e in self._load_index() if e.get("session_id") != session_id]
        self._save_index(entries)

    def list_sessions(self, work_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出会话索引条目.

        Args:
            work_dir: 若提供，仅返回该工作目录下的会话.

        """
        entries = self._load_index()
        if work_dir:
            key = _work_dir_key(work_dir)
            entries = [e for e in entries if e.get("work_dir_key") == key]
        # 按 updated_at 倒序
        entries.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return entries

    # ------------------------------------------------------------------ #
    # 路径构建
    # ------------------------------------------------------------------ #

    def session_dir(self, work_dir: str, session_id: str) -> Path:
        """获取会话存储目录."""
        return self.root / "sessions" / _work_dir_key(work_dir) / session_id

    def meta_path(self, work_dir: str, session_id: str) -> Path:
        return self.session_dir(work_dir, session_id) / "meta.json"

    def state_path(self, work_dir: str, session_id: str) -> Path:
        return self.session_dir(work_dir, session_id) / "state.json"

    def wire_path(self, work_dir: str, session_id: str, agent_id: str = "main") -> Path:
        """获取指定 Agent 的 wire 文件路径."""
        return self.session_dir(work_dir, session_id) / "agents" / agent_id / "wire.jsonl"

    # ------------------------------------------------------------------ #
    # 元数据
    # ------------------------------------------------------------------ #

    def save_meta(
        self,
        work_dir: str,
        session_id: str,
        meta: Dict[str, Any],
    ) -> None:
        """保存会话元数据."""
        path = self.meta_path(work_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock:
                path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"保存会话元数据失败 [{session_id}]: {e}")

    def load_meta(self, work_dir: str, session_id: str) -> Optional[Dict[str, Any]]:
        """加载会话元数据."""
        path = self.meta_path(work_dir, session_id)
        if not path.exists():
            return None
        try:
            with self._lock:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"加载会话元数据失败 [{session_id}]: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #

    def save_state(self, work_dir: str, session_id: str, state_text: str) -> None:
        """保存 AgentState JSON 文本."""
        path = self.state_path(work_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock:
                path.write_text(state_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"保存会话状态失败 [{session_id}]: {e}")

    def load_state(self, work_dir: str, session_id: str) -> Optional[str]:
        """加载 AgentState JSON 文本."""
        path = self.state_path(work_dir, session_id)
        if not path.exists():
            return None
        try:
            with self._lock:
                return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"加载会话状态失败 [{session_id}]: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Wire（交互记录）
    # ------------------------------------------------------------------ #

    def append_wire(
        self,
        work_dir: str,
        session_id: str,
        event: Dict[str, Any],
        agent_id: str = "main",
    ) -> None:
        """追加一条交互记录到 wire.jsonl."""
        path = self.wire_path(work_dir, session_id, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock, path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"追加 wire 记录失败 [{session_id}/{agent_id}]: {e}")

    def load_wire(
        self,
        work_dir: str,
        session_id: str,
        agent_id: str = "main",
    ) -> List[Dict[str, Any]]:
        """加载 wire 记录列表."""
        path = self.wire_path(work_dir, session_id, agent_id)
        if not path.exists():
            return []
        try:
            entries = []
            with self._lock, path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
        except Exception as e:
            logger.warning(f"加载 wire 记录失败 [{session_id}/{agent_id}]: {e}")
            return []

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #

    def delete_session(self, work_dir: str, session_id: str) -> bool:
        """删除会话的所有持久化数据."""
        import shutil
        path = self.session_dir(work_dir, session_id)
        if not path.exists():
            return False
        try:
            shutil.rmtree(path)
            self.remove_index(session_id)
            return True
        except Exception as e:
            logger.warning(f"删除会话数据失败 [{session_id}]: {e}")
            return False

    def cleanup_orphaned(self, valid_session_ids: List[str]) -> int:
        """清理索引中不存在但磁盘上残留的会话目录.

        Returns:
            清理的目录数量.

        """
        import shutil
        valid_set = set(valid_session_ids)
        count = 0
        sessions_root = self.root / "sessions"
        if not sessions_root.exists():
            return 0
        for work_dir_key_dir in sessions_root.iterdir():
            if not work_dir_key_dir.is_dir():
                continue
            for session_dir in work_dir_key_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                sid = session_dir.name
                if sid not in valid_set:
                    try:
                        shutil.rmtree(session_dir)
                        count += 1
                        logger.info(f"清理孤儿会话目录: {sid}")
                    except Exception as e:
                        logger.warning(f"清理孤儿目录失败 [{sid}]: {e}")
        return count
