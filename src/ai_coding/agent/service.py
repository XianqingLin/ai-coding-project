"""Agent 对外统一服务接口.

AgentService 是 CLI、TUI、Web UI、测试脚本与 Agent 核心之间的薄封装层。
它隐藏了 SessionManager、LLM factory、tools 等内部依赖，提供统一、
可编程的会话管理和消息发送接口。
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ai_coding.agent.core import LangGraphAgent
from ai_coding.agent.events import (
    AgentEvent,
    AssistantChunkEvent,
    AssistantEndEvent,
    AssistantStartEvent,
    ErrorEvent,
    ObservationEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEvent,
    UserInputEvent,
)
from ai_coding.agent.session import SessionManager
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.environment import collect_environment_info
from ai_coding.llm import create_lc_llm
from ai_coding.prompts import PromptContext
from ai_coding.logger import get_logger
from ai_coding.tools import create_default_tools

logger = get_logger(__name__)


class AgentService:
    """Agent 统一服务接口.

    用法示例:
        svc = AgentService(work_dir="./workspace", auto_approve=True)
        svc.create_session("my-task")
        reply = svc.send_message("帮我查看 README.md")

        # 流式消费事件（适合 TUI / Web UI）
        for event in svc.send_message_stream("实现一个俄罗斯方块"):
            print(event)
    """

    def __init__(
        self,
        work_dir: str,
        llm_provider: Optional[str] = None,
        auto_approve: bool = False,
        llm_factory: Optional[Callable[[], Any]] = None,
        enable_env_info: bool = True,
    ) -> None:
        """初始化 AgentService.

        Args:
            work_dir: Agent 的工作目录.
            llm_provider: LLM 提供商名称，默认使用配置中的 DEFAULT_LLM_PROVIDER.
            auto_approve: 是否自动批准工具调用.
            llm_factory: 可选的 LLM 工厂函数，主要用于测试注入 Mock LLM.
            enable_env_info: 是否在系统提示词中注入环境信息.
        """
        self.work_dir = str(Path(work_dir).expanduser().resolve())
        self.llm_provider = llm_provider or DEFAULT_LLM_PROVIDER
        self.auto_approve = auto_approve
        self._llm_factory = llm_factory or (lambda: create_lc_llm(self.llm_provider))

        prompt_context = None
        if enable_env_info:
            environment_info = collect_environment_info(self.work_dir, self.llm_provider)
            prompt_context = PromptContext(environment_info=environment_info)

        self._sm = SessionManager(
            llm_factory=self._llm_factory,
            tools_factory=create_default_tools,
            auto_approve=self.auto_approve,
            work_dir=self.work_dir,
            prompt_context=prompt_context,
        )

    # ------------------------------------------------------------------ #
    # 会话管理
    # ------------------------------------------------------------------ #

    def create_session(self, name: str = "") -> str:
        """创建新会话.

        Args:
            name: 会话名称，为空时自动生成.

        Returns:
            新会话 ID.
        """
        return self._sm.create(name=name)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话."""
        return self._sm.list()

    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话."""
        return self._sm.switch(session_id)

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话."""
        return self._sm.delete(session_id)

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """重命名会话."""
        return self._sm.rename(session_id, new_name)

    @property
    def current_session_id(self) -> Optional[str]:
        """当前活跃会话 ID."""
        return self._sm.current_session_id

    def _resolve_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """解析目标会话 ID，默认使用当前会话."""
        if session_id is None:
            return self._sm.current_session_id
        return session_id

    def _get_agent(self, session_id: Optional[str] = None) -> Optional[LangGraphAgent]:
        """获取指定会话的 Agent 实例（写操作：会隐式切换当前会话）."""
        sid = self._resolve_session_id(session_id)
        if sid is None:
            return None
        if sid == self._sm.current_session_id:
            return self._sm.get_current_agent()
        # 非当前会话：先切换再获取
        if not self._sm.switch(sid):
            return None
        return self._sm.get_current_agent()

    def _get_agent_safe(self, session_id: Optional[str] = None) -> Optional[LangGraphAgent]:
        """获取指定会话的 Agent 实例（只读：无副作用，不切换当前会话）."""
        sid = self._resolve_session_id(session_id)
        if sid is None:
            return None
        if sid == self._sm.current_session_id:
            return self._sm.get_current_agent()
        return self._sm.get_agent(sid)

    # ------------------------------------------------------------------ #
    # 消息发送
    # ------------------------------------------------------------------ #

    def send_message(self, text: str, session_id: Optional[str] = None) -> str:
        """同步发送消息并返回 Agent 最终回复.

        Args:
            text: 用户输入文本.
            session_id: 目标会话 ID，默认当前会话.

        Returns:
            Agent 最终文本回复.
        """
        agent = self._get_agent(session_id)
        if agent is None:
            logger.error("send_message: 没有可用的 Agent 会话")
            return "[错误] 没有可用的 Agent 会话"
        return agent.run(text)

    def send_message_stream(
        self, text: str, session_id: Optional[str] = None
    ) -> Iterator[AgentEvent]:
        """流式发送消息，返回标准化事件流.

        Args:
            text: 用户输入文本.
            session_id: 目标会话 ID，默认当前会话.

        Yields:
            AgentEvent 事件对象.
        """
        yield UserInputEvent(text=text)

        agent = self._get_agent(session_id)
        if agent is None:
            logger.error("send_message_stream: 没有可用的 Agent 会话")
            yield ErrorEvent(text="[错误] 没有可用的 Agent 会话")
            return

        if not hasattr(agent, "run_stream_verbose"):
            yield ErrorEvent(text="[错误] Agent 不支持流式事件模式")
            return

        try:
            for raw in agent.run_stream_verbose(text):
                event = self._map_raw_event(raw)
                if event is not None:
                    yield event
        except Exception as e:
            logger.error(f"send_message_stream 失败: {e}", exc_info=True)
            yield ErrorEvent(text=f"[错误] 流式运行失败: {e}")

    def _map_raw_event(self, raw: Dict[str, Any]) -> Optional[AgentEvent]:
        """把 LangGraphAgent 的原始事件映射为标准 AgentEvent."""
        etype = raw.get("type")
        if etype == "user_input":
            return UserInputEvent(text=raw.get("input", ""))
        if etype == "thinking_start":
            return ThinkingStartEvent()
        if etype == "thinking_chunk":
            return ThinkingChunkEvent(text=raw.get("text", ""))
        if etype == "thinking_end":
            return ThinkingEndEvent()
        if etype == "assistant_start":
            return AssistantStartEvent()
        if etype == "assistant_chunk":
            return AssistantChunkEvent(text=raw.get("text", ""))
        if etype == "assistant_end":
            return AssistantEndEvent(text=raw.get("text", ""))
        if etype == "tool_call":
            return ToolCallEvent(
                name=raw.get("name", ""),
                args=raw.get("args", {}),
            )
        if etype == "observation":
            return ObservationEvent(text=raw.get("text", ""))
        if etype == "error":
            return ErrorEvent(text=raw.get("text", ""))
        logger.debug(f"未知事件类型: {etype}")
        return None

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #

    def get_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取指定会话的对话历史（只读，不切换当前会话）."""
        agent = self._get_agent_safe(session_id)
        if agent is None:
            return []
        return agent.get_history()

    def get_context_usage(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取指定会话的上下文窗口使用情况（只读，不切换当前会话）."""
        agent = self._get_agent_safe(session_id)
        if agent is None:
            return {"used_tokens": 0, "limit_tokens": 0, "percentage": 0.0}
        return agent.get_context_usage()

    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取 Agent 运行统计信息（只读，不切换当前会话）."""
        agent = self._get_agent_safe(session_id)
        if agent is None:
            return {}
        return agent.get_stats()

    def get_system_prompt(self, session_id: Optional[str] = None) -> str:
        """获取指定会话当前使用的 system prompt（只读，不切换当前会话）."""
        agent = self._get_agent_safe(session_id)
        return agent.system_prompt if agent else ""

    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """获取当前会话的摘要信息（名称、ID、消息数等）."""
        sessions = self.list_sessions()
        current_id = self.current_session_id
        for s in sessions:
            if s.get("session_id") == current_id:
                return s
        return None
