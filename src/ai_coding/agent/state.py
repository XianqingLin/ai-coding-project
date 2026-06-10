"""LangGraph 状态定义.

AgentState 是自我管理容量的短期记忆容器：
- messages 由 LangGraph 的 add_messages reducer 累积，由 LangGraphAgent 显式管理容量（compact）
- file_snapshots 由 tools_node 动态更新，跨轮次保留
"""

from typing import Annotated, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def merge_file_snapshots(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    """合并文件快照：右侧覆盖左侧."""
    merged = dict(left)
    merged.update(right)
    return merged


def replace_todos(left: List[dict], right: List[dict]) -> List[dict]:
    """替换 todo 列表：右侧非空则直接覆盖."""
    return list(right) if right is not None else list(left)


def merge_approved_tools(left: List[str], right: List[str]) -> List[str]:
    """合并已授权工具列表：取并集."""
    return list(set(left) | set(right))


def replace_background_tasks(left: List[dict], right: List[dict]) -> List[dict]:
    """替换后台任务列表：右侧非空则直接覆盖."""
    return list(right) if right is not None else list(left)


def replace_plan_mode(left: bool, right: bool) -> bool:
    """替换 Plan 模式状态：右侧非 None 则覆盖."""
    return bool(right) if right is not None else bool(left)


def replace_plan_file_path(left: str, right: str) -> str:
    """替换计划文件路径：右侧非 None 则覆盖."""
    return str(right) if right is not None else str(left)


class AgentState(TypedDict):
    """Agent 图状态 —— 自我管理容量的短期记忆容器.

    - messages: 对话历史（LangGraph 追加 reducer）。由 LangGraphAgent.compact()
      显式压缩后替换，确保体积可控。llm_node 直接消费，不再二次压缩。
    - file_snapshots: 当前最新文件内容快照（工具执行后更新，LLM 调用前注入）
    - todos: 当前任务列表（工具执行后更新，LLM 调用前注入）
    - background_tasks: 后台任务状态列表（工具执行后更新）
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    file_snapshots: Annotated[Dict[str, str], merge_file_snapshots]
    todos: Annotated[List[dict], replace_todos]
    globally_approved_tools: Annotated[List[str], merge_approved_tools]
    background_tasks: Annotated[List[dict], replace_background_tasks]
    plan_mode: Annotated[bool, replace_plan_mode]
    plan_file_path: Annotated[str, replace_plan_file_path]
