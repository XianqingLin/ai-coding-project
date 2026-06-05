"""会话管理模块.

提供多会话的创建、切换、列表和删除功能.
每个会话拥有独立的 LangGraphAgent 实例和对话上下文.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ai_coding.agent.core import LangGraphAgent
from ai_coding.logger import get_logger

logger = get_logger(__name__)


class Session:
    """单个会话的数据模型."""

    def __init__(
        self,
        session_id: str,
        name: str,
        created_at: float,
        agent: LangGraphAgent,
    ) -> None:
        self.session_id = session_id
        self.name = name
        self.created_at = created_at
        self.agent = agent

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "message_count": len(self.agent.get_history()),
            "thread_id": self.agent.thread_id,
        }


class SessionManager:
    """会话管理器.

    管理多个并发的 Agent 会话，支持创建、切换、列表、删除.
    会话元数据（名称、创建时间等）持久化到 JSON 文件；
    Agent 对话状态由 MemorySaver 在内存中维护——进程重启后
    会话列表保留，但各会话的上下文历史需要重新建立.

    Attributes:
        storage_path: 会话元数据存储路径.
        sessions: 会话字典 (session_id -> Session).
        current_session_id: 当前活跃会话 ID.

    """

    def __init__(
        self,
        llm_factory: Callable[[], Any],
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        storage_path: str = ".sessions/sessions.json",
    ) -> None:
        self.llm_factory = llm_factory
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.storage_path = Path(storage_path)
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None

        self._load_metadata()

        # 如果没有任何会话，自动创建一个默认会话
        if not self.sessions:
            self.create(name="default")

    def _load_metadata(self) -> None:
        """从文件加载会话元数据."""
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for sid, meta in data.get("sessions", {}).items():
                # 恢复会话：重建 Agent（MemorySaver 为空，上下文需重新建立）
                thread_id = meta.get("thread_id")
                agent = self._create_agent(thread_id=thread_id)
                self.sessions[sid] = Session(
                    session_id=sid,
                    name=meta.get("name", sid),
                    created_at=meta.get("created_at", time.time()),
                    agent=agent,
                )
            self.current_session_id = data.get("current_session_id")
            logger.info(f"已加载 {len(self.sessions)} 个会话")
        except Exception as e:
            logger.warning(f"加载会话元数据失败: {e}")

    def _save_metadata(self) -> None:
        """保存会话元数据到文件."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "sessions": {
                    sid: session.to_dict()
                    for sid, session in self.sessions.items()
                },
                "current_session_id": self.current_session_id,
            }
            self.storage_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"保存会话元数据失败: {e}")

    def _create_agent(self, thread_id: Optional[str] = None) -> LangGraphAgent:
        """创建新的 Agent 实例."""
        return LangGraphAgent(
            llm=self.llm_factory(),
            tools=self.tools,
            system_prompt=self.system_prompt,
            thread_id=thread_id,
            max_iterations=10,
            streaming=True,
        )

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
        self.sessions[sid] = Session(
            session_id=sid,
            name=name,
            created_at=time.time(),
            agent=agent,
        )
        self.current_session_id = sid
        self._save_metadata()
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
        self._save_metadata()
        session = self.sessions[session_id]
        logger.info(f"切换会话: {session.name} ({session_id})")
        return True

    def delete(self, session_id: str) -> bool:
        """删除会话.

        Args:
            session_id: 要删除的会话 ID.

        Returns:
            是否删除成功.

        """
        if session_id not in self.sessions:
            return False
        name = self.sessions[session_id].name
        del self.sessions[session_id]
        if self.current_session_id == session_id:
            self.current_session_id = next(iter(self.sessions.keys()), None)
        self._save_metadata()
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
        self._save_metadata()
        return True
