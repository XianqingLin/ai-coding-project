"""会话持久化模块.

提供 AgentState 的序列化、存储引擎和会话恢复能力.
"""

from ai_coding.persistence.config import get_config, get_storage_root
from ai_coding.persistence.serializer import (
    deserialize_messages,
    deserialize_state,
    serialize_messages,
    serialize_state,
    state_from_json,
    state_to_json,
)
from ai_coding.persistence.storage import StorageEngine, _work_dir_key

__all__ = [
    "StorageEngine",
    "_work_dir_key",
    "serialize_state",
    "deserialize_state",
    "serialize_messages",
    "deserialize_messages",
    "state_to_json",
    "state_from_json",
    "get_config",
    "get_storage_root",
]
