from __future__ import annotations

"""子 Agent 管理模块.

提供 SubAgentManager 来创建、管理和调度子 Agent 实例。
子 Agent 拥有独立的 LangGraphAgent 实例和上下文窗口。
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ai_coding.logger import get_logger

if TYPE_CHECKING:
    from ai_coding.agent.core import LangGraphAgent
    from ai_coding.tools.base import Tool

logger = get_logger(__name__)

# 子 Agent 同步模式超时（秒）
SUB_AGENT_SYNC_TIMEOUT = 1800  # 30 分钟

# 子 Agent 最大迭代次数
SUB_AGENT_MAX_ITERATIONS = 10

# 子 Agent 类型 → 可用工具名
_SUB_AGENT_TOOL_SETS = {
    "coder": None,  # None 表示使用全部工具
    "explore": ["read_file", "list_dir", "grep", "glob"],
    "plan": ["read_file", "list_dir", "grep", "glob"],
}

_SUB_AGENT_PROMPTS = {
    "coder": None,  # None 表示使用默认系统提示
    "explore": (
        "你是一个代码库探索助手。你的任务是在不修改任何文件的前提下，"
        "搜索、阅读和理解代码，然后向主 Agent 返回清晰、结构化的总结。\n\n"
        "你只能使用只读工具（read_file, list_dir, grep, glob）。"
        "不要尝试写入、编辑文件或执行命令。"
    ),
    "plan": (
        "你是一个架构设计与实现规划助手。你的任务是分析代码库，"
        "设计清晰的实现方案，并返回详细的执行计划。\n\n"
        "你只能使用只读工具（read_file, list_dir, grep, glob）来了解代码结构。"
        "不要执行命令或修改文件。你的输出应该是可直接指导 coder 子 Agent 执行的文本计划。"
    ),
}


@dataclass
class SubAgentInstance:
    """子 Agent 实例数据模型."""

    instance_id: str
    agent_type: str
    task: str
    status: str = "running"  # running, completed, failed
    result: str = ""
    agent: Optional[LangGraphAgent] = None
    notified: bool = False
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class SubAgentManager:
    """子 Agent 管理器（单例模式）.

    负责创建、调度和管理所有子 Agent 实例。
    """

    _instance: Optional["SubAgentManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SubAgentManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._agents: Dict[str, SubAgentInstance] = {}
        self._lock = threading.Lock()

    def dispatch(
        self,
        agent_type: str,
        prompt: str,
        llm: Any,
        llm_factory: Optional[Callable[[], Any]] = None,
        run_in_background: bool = False,
        instance_id: Optional[str] = None,
    ) -> str:
        """派发子 Agent 任务.

        Args:
            agent_type: 子 Agent 类型 (coder / explore / plan)
            prompt: 任务描述
            llm: 主 Agent 的 LLM 实例（作为备选）
            llm_factory: LLM 工厂函数，用于创建独立 LLM 实例
            run_in_background: 是否后台运行
            instance_id: 可选，唤回已有实例

        Returns:
            运行结果（同步模式）或任务 ID（后台模式）
        """
        if instance_id and instance_id in self._agents:
            return self._resume_instance(instance_id, prompt, llm, llm_factory, run_in_background)

        # 创建新实例
        sid = f"sub_{uuid.uuid4().hex[:8]}"
        tools = self._get_tools_for_type(agent_type)
        system_prompt = self._get_prompt_for_type(agent_type)

        # 创建 LLM 实例
        sub_llm = llm_factory() if llm_factory else llm

        # 延迟导入避免循环依赖
        from ai_coding.agent.core import LangGraphAgent
        agent = LangGraphAgent(
            llm=sub_llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=SUB_AGENT_MAX_ITERATIONS,
            streaming=False,
            auto_approve=True,
            thread_id=sid,
        )

        instance = SubAgentInstance(
            instance_id=sid,
            agent_type=agent_type,
            task=prompt,
            agent=agent,
        )

        with self._lock:
            self._agents[sid] = instance

        if run_in_background:
            thread = threading.Thread(
                target=self._run_agent,
                args=(sid,),
                daemon=True,
            )
            thread.start()
            logger.info(f"[SubAgent] 后台启动 {sid} ({agent_type})")
            return (
                f"[子 Agent 后台启动] ID: {sid}\n"
                f"类型: {agent_type}\n"
                f"任务: {prompt[:200]}\n"
                f"完成后结果将自动回到主 Agent。"
            )

        # 同步运行（带 30 分钟超时）
        thread = threading.Thread(
            target=self._run_agent,
            args=(sid,),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=SUB_AGENT_SYNC_TIMEOUT)
        if thread.is_alive():
            with self._lock:
                self._agents[sid].status = "failed"
                self._agents[sid].result = f"[错误] 子 Agent 执行超时（{SUB_AGENT_SYNC_TIMEOUT // 60} 分钟）"
            logger.warning(f"[SubAgent] {sid} 执行超时（{SUB_AGENT_SYNC_TIMEOUT // 60} 分钟）")

        with self._lock:
            inst = self._agents[sid]
        return inst.result

    def _run_agent(self, instance_id: str) -> None:
        """在指定实例上运行子 Agent（线程安全）."""
        with self._lock:
            instance = self._agents.get(instance_id)
        if not instance or instance.agent is None:
            logger.warning(f"[SubAgent] 实例不存在或 agent 为空: {instance_id}")
            return

        try:
            result = instance.agent.run(instance.task)
            status = "completed"
        except Exception as e:
            result = f"[错误] 子 Agent 执行失败: {e}"
            status = "failed"
            logger.error(f"[SubAgent] {instance_id} 执行失败: {e}", exc_info=True)

        with self._lock:
            self._agents[instance_id].status = status
            self._agents[instance_id].result = result
            self._agents[instance_id].completed_at = time.time()

        logger.info(f"[SubAgent] {instance_id} 完成 | 状态={status} | 结果长度={len(result)}")

    def _resume_instance(
        self,
        instance_id: str,
        prompt: str,
        llm: Any,
        llm_factory: Optional[Callable[[], Any]],
        run_in_background: bool,
    ) -> str:
        """唤回已有实例继续运行."""
        with self._lock:
            instance = self._agents.get(instance_id)
        if not instance:
            return f"[错误] 实例不存在: {instance_id}"

        if instance.status == "running":
            return f"[错误] 实例 {instance_id} 仍在运行中，请等待完成"

        # 更新任务和状态
        instance.task = prompt
        instance.status = "running"
        instance.result = ""
        instance.notified = False
        instance.completed_at = None

        if instance.agent is None:
            return f"[错误] 实例 {instance_id} 的 agent 已丢失，无法恢复"

        # 如果提供了新的 LLM，更新 agent
        new_llm = llm_factory() if llm_factory else llm
        if new_llm and new_llm is not instance.agent.llm:
            from ai_coding.agent.core import LangGraphAgent
            old_state = instance.agent.state
            new_agent = LangGraphAgent(
                llm=new_llm,
                tools=instance.agent.tools,
                system_prompt=instance.agent.system_prompt,
                max_iterations=instance.agent.max_iterations,
                streaming=False,
                auto_approve=True,
                thread_id=instance_id,
            )
            if old_state is not None:
                new_agent.state = old_state
            instance.agent = new_agent

        # 追加新任务到子 Agent 的 state
        if instance.agent.state is not None:
            from langchain_core.messages import HumanMessage
            messages = list(instance.agent.state.get("messages", []))
            messages.append(HumanMessage(content=prompt))
            instance.agent.state["messages"] = messages

        if run_in_background:
            thread = threading.Thread(
                target=self._run_agent,
                args=(instance_id,),
                daemon=True,
            )
            thread.start()
            return (
                f"[子 Agent 唤回并后台运行] ID: {instance_id}\n"
                f"类型: {instance.agent_type}\n"
                f"新任务: {prompt[:200]}"
            )

        # 同步运行（带 30 分钟超时）
        thread = threading.Thread(
            target=self._run_agent,
            args=(instance_id,),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=SUB_AGENT_SYNC_TIMEOUT)
        if thread.is_alive():
            with self._lock:
                self._agents[instance_id].status = "failed"
                self._agents[instance_id].result = f"[错误] 子 Agent 执行超时（{SUB_AGENT_SYNC_TIMEOUT // 60} 分钟）"
            logger.warning(f"[SubAgent] {instance_id} 执行超时（{SUB_AGENT_SYNC_TIMEOUT // 60} 分钟）")

        return instance.result

    def get_instance(self, instance_id: str) -> Optional[SubAgentInstance]:
        """获取指定实例."""
        with self._lock:
            return self._agents.get(instance_id)

    def list_instances(self, status: Optional[str] = None) -> List[SubAgentInstance]:
        """列出所有实例，可按状态过滤."""
        with self._lock:
            instances = list(self._agents.values())
        if status:
            instances = [i for i in instances if i.status == status]
        return instances

    def get_pending_notifications(self) -> List[SubAgentInstance]:
        """获取已完成但未通知主 Agent 的实例列表."""
        with self._lock:
            return [
                i for i in self._agents.values()
                if i.status in ("completed", "failed") and not i.notified
            ]

    def mark_notified(self, instance_id: str) -> None:
        """标记实例为已通知."""
        with self._lock:
            if instance_id in self._agents:
                self._agents[instance_id].notified = True

    def to_dict(self, instance: SubAgentInstance) -> dict:
        """将实例序列化为字典."""
        return {
            "instance_id": instance.instance_id,
            "agent_type": instance.agent_type,
            "task": instance.task,
            "status": instance.status,
            "result": instance.result,
            "notified": instance.notified,
            "created_at": instance.created_at,
            "completed_at": instance.completed_at,
        }

    @staticmethod
    def _get_tools_for_type(agent_type: str) -> List[Tool]:
        """根据类型返回可用工具列表."""
        from ai_coding.tools import DEFAULT_TOOLS
        tool_names = _SUB_AGENT_TOOL_SETS.get(agent_type)
        if tool_names is None:
            return list(DEFAULT_TOOLS)
        all_tools = {t.name: t for t in DEFAULT_TOOLS}
        return [all_tools[n] for n in tool_names if n in all_tools]

    @staticmethod
    def _get_prompt_for_type(agent_type: str) -> Optional[str]:
        """根据类型返回专用 system_prompt."""
        return _SUB_AGENT_PROMPTS.get(agent_type)
