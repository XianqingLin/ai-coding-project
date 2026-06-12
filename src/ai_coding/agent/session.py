"""会话管理模块.

提供多会话的创建、切换、列表和删除功能.
每个会话拥有独立的 LangGraphAgent 实例和对话上下文.
支持完整的状态持久化：AgentState、消息历史、交互记录.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ai_coding.agent.core import LangGraphAgent
from ai_coding.logger import get_logger
from ai_coding.persistence import StorageEngine, state_from_json, state_to_json

logger = get_logger(__name__)


class Session:
    """单个会话的数据模型."""

    def __init__(
        self,
        session_id: str,
        name: str,
        created_at: float,
        agent: LangGraphAgent,
        work_dir: str = "",
    ) -> None:
        self.session_id = session_id
        self.name = name
        self.created_at = created_at
        self.agent = agent
        self.work_dir = work_dir

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "message_count": len(self.agent.get_history()),
            "thread_id": self.agent.thread_id,
            "work_dir": self.work_dir,
        }


class SessionManager:
    """会话管理器.

    管理多个并发的 Agent 会话，支持创建、切换、列表、删除.
    会话元数据和 AgentState 均持久化到 ~/.ai-coding/；
    进程重启后可恢复完整的对话上下文.

    Attributes:
        work_dir: 当前工作目录（用于隔离不同项目的会话）.
        storage: 持久化存储引擎.
        sessions: 会话字典 (session_id -> Session).
        current_session_id: 当前活跃会话 ID.

    """

    def __init__(
        self,
        llm_factory: Callable[[], Any],
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        auto_approve: bool = False,
        work_dir: Optional[str] = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.auto_approve = auto_approve
        self.work_dir = str(Path(work_dir).resolve()) if work_dir else str(Path.cwd().resolve())

        self.storage = StorageEngine()
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None

        self._load_sessions()

        # 如果没有任何会话，自动创建一个默认会话
        if not self.sessions:
            self.create(name="default")

    def _load_sessions(self) -> None:
        """从持久化存储恢复会话.

        恢复流程：
        1. 读取 session_index，筛选当前 work_dir 的条目
        2. 对每个条目，读取 meta.json 和 state.json
        3. 重建 LangGraphAgent，并注入恢复的 AgentState
        """
        entries = self.storage.list_sessions(self.work_dir)
        for entry in entries:
            sid = entry.get("session_id")
            if not sid:
                continue

            meta = self.storage.load_meta(self.work_dir, sid)
            if meta is None:
                continue

            thread_id = meta.get("thread_id")
            agent = self._create_agent(thread_id=thread_id)

            # 尝试恢复 AgentState
            state_text = self.storage.load_state(self.work_dir, sid)
            if state_text:
                try:
                    agent.state = state_from_json(state_text)
                    logger.info(f"会话 [{sid}] AgentState 已恢复")
                except Exception as e:
                    logger.warning(f"会话 [{sid}] AgentState 恢复失败: {e}")

            session = Session(
                session_id=sid,
                name=meta.get("name", sid),
                created_at=meta.get("created_at", time.time()),
                agent=agent,
                work_dir=self.work_dir,
            )
            self.sessions[sid] = session

        # 设置当前会话：优先使用索引中标记的，否则取第一个
        if self.sessions:
            # 查找索引中是否有当前会话标记
            for entry in entries:
                if entry.get("is_current"):
                    sid = entry.get("session_id")
                    if sid in self.sessions:
                        self.current_session_id = sid
                        break
            if not self.current_session_id:
                self.current_session_id = next(iter(self.sessions.keys()))
            logger.info(f"已加载 {len(self.sessions)} 个会话")

    def _save_session_meta(self, session: Session, is_current: bool = False) -> None:
        """保存单个会话的元数据和索引."""
        meta = session.to_dict()
        self.storage.save_meta(self.work_dir, session.session_id, meta)
        self.storage.upsert_index(
            session_id=session.session_id,
            work_dir=self.work_dir,
            name=session.name,
            created_at=session.created_at,
            updated_at=time.time(),
        )
        # 更新索引中的 is_current 标记
        if is_current:
            self._mark_current_in_index(session.session_id)

    def _mark_current_in_index(self, session_id: str) -> None:
        """在索引中标记当前会话."""
        entries = self.storage.list_sessions()
        for entry in entries:
            entry["is_current"] = (entry.get("session_id") == session_id)
        # 重写索引
        import json as _json
        try:
            path = self.storage.index_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"更新当前会话标记失败: {e}")

    def _create_agent(self, thread_id: Optional[str] = None) -> LangGraphAgent:
        """创建新的 Agent 实例."""
        return LangGraphAgent(
            llm=self.llm_factory(),
            tools=self.tools,
            system_prompt=self.system_prompt,
            thread_id=thread_id,
            max_iterations=10,
            streaming=True,
            auto_approve=self.auto_approve,
            llm_factory=self.llm_factory,
            on_state_change=lambda: self.save_state(),
        )

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #

    def create(self, name: str = "") -> str:
        """创建新会话.

        Args:
            name: 会话名称. 为空时自动生成.

        Returns:
            新会话 ID.

        """
        sid = uuid.uuid4().hex[:8]
        name = name.strip() or f"session-{sid}"
        agent = self._create_agent()
        session = Session(
            session_id=sid,
            name=name,
            created_at=time.time(),
            agent=agent,
            work_dir=self.work_dir,
        )
        self.sessions[sid] = session
        self.current_session_id = sid
        self._save_session_meta(session, is_current=True)
        logger.info(f"创建会话: {name} ({sid})")
        return sid

    def switch(self, session_id: str) -> bool:
        """切换到指定会话.

        Args:
            session_id: 目标会话 ID.

        Returns:
            是否切换成功.

        """
        if session_id not in self.sessions:
            logger.warning(f"会话不存在: {session_id}")
            return False
        self.current_session_id = session_id
        self._mark_current_in_index(session_id)
        session = self.sessions[session_id]
        logger.info(f"切换会话: {session.name} ({session_id})")
        return True

    def delete(self, session_id: str) -> bool:
        """删除会话.

        如果删除的是最后一个会话，会自动创建一个默认会话以保证
        始终存在可用的当前会话.

        Args:
            session_id: 要删除的会话 ID.

        Returns:
            是否删除成功.

        """
        if session_id not in self.sessions:
            return False
        name = self.sessions[session_id].name
        was_current = self.current_session_id == session_id
        del self.sessions[session_id]
        self.storage.delete_session(self.work_dir, session_id)
        if was_current:
            self.current_session_id = next(iter(self.sessions.keys()), None)
            if self.current_session_id:
                self._mark_current_in_index(self.current_session_id)
        # 删除后如果没有剩余会话，自动创建默认会话
        if not self.sessions:
            self.create(name="default")
            logger.info("所有会话已删除，自动创建默认会话")
        logger.info(f"删除会话: {name} ({session_id})")
        return True

    def list(self) -> List[dict]:
        """列出所有会话.

        Returns:
            会话信息字典列表，按创建时间排序.

        """
        result = []
        for sid, session in self.sessions.items():
            d = session.to_dict()
            d["is_current"] = sid == self.current_session_id
            result.append(d)
        result.sort(key=lambda x: x["created_at"])
        return result

    @property
    def current(self) -> Optional[Session]:
        """获取当前会话."""
        if not self.current_session_id:
            return None
        return self.sessions.get(self.current_session_id)

    def get_current_agent(self) -> Optional[LangGraphAgent]:
        """获取当前会话的 Agent 实例."""
        session = self.current
        return session.agent if session else None

    def rename(self, session_id: str, new_name: str) -> bool:
        """重命名会话.

        Args:
            session_id: 目标会话 ID.
            new_name: 新名称.

        Returns:
            是否重命名成功.

        """
        if session_id not in self.sessions:
            return False
        self.sessions[session_id].name = new_name.strip() or self.sessions[session_id].name
        self._save_session_meta(self.sessions[session_id], is_current=(session_id == self.current_session_id))
        return True

    # ------------------------------------------------------------------ #
    # 状态持久化
    # ------------------------------------------------------------------ #

    def save_state(self, session_id: Optional[str] = None) -> bool:
        """保存指定会话的完整 AgentState.

        Args:
            session_id: 若为空，保存当前会话.

        Returns:
            是否保存成功.

        """
        sid = session_id or self.current_session_id
        if not sid or sid not in self.sessions:
            return False
        session = self.sessions[sid]
        try:
            state_text = state_to_json(session.agent.state)
            self.storage.save_state(self.work_dir, sid, state_text)
            self._save_session_meta(session, is_current=(sid == self.current_session_id))
            return True
        except Exception as e:
            logger.warning(f"保存会话状态失败 [{sid}]: {e}")
            return False

    def append_wire(
        self,
        event: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """追加交互记录到当前会话.

        Args:
            event: 交互事件字典.
            session_id: 目标会话 ID，默认当前会话.
            agent_id: Agent 标识，主 Agent 为 "main".

        """
        sid = session_id or self.current_session_id
        if not sid:
            return
        self.storage.append_wire(self.work_dir, sid, event, agent_id)
