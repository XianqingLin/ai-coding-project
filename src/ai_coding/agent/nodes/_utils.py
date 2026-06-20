"""Agent 节点内部共享工具函数."""

from typing import Any, Dict, cast


def normalize_tool_args(args: Any) -> Dict[str, Any]:
    """将工具参数统一归一化为普通字典（兼容 Pydantic v1/v2、dict、其他可映射对象）.

    Args:
        args: LLM 或工具调用传入的参数对象。

    Returns:
        普通 dict。无法转换时回退到 {"_raw": str(args)}。
    """
    if args is None:
        return {}
    if hasattr(args, "model_dump"):
        return cast(Dict[str, Any], args.model_dump())
    if hasattr(args, "dict"):
        return cast(Dict[str, Any], args.dict())
    if isinstance(args, dict):
        return args
    try:
        return dict(args)
    except Exception:
        return {"_raw": str(args)}
