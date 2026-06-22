"""MemoryExtractor 单元测试."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from ai_coding.memory.extractor import MemoryExtractor
from ai_coding.memory.models import MemoryScope, MemoryType
from ai_coding.memory.store import MemoryStore
from ai_coding.mock_llm import MockChatModel, mock_text


def _make_extractor(response_text: str) -> MemoryExtractor:
    """使用 Mock LLM 创建 MemoryExtractor."""
    return MemoryExtractor(
        llm_factory=lambda: MockChatModel(responses=[mock_text(response_text)])
    )


class TestMemoryExtractorParsing:
    def test_extract_valid_memories(self) -> None:
        response = (
            '{"memories": [{'
            '"content": "使用中文回复", '
            '"type": "preference", '
            '"scope": "project", '
            '"confidence": 0.9, '
            '"explicit": true'
            "}]}"
        )
        extractor = _make_extractor(response)
        messages = [
            HumanMessage(content="以后都用中文回复我"),
            AIMessage(content="好的，我会使用中文回复。"),
        ]
        entries = extractor.extract(messages, session_id="sess_001")

        assert len(entries) == 1
        assert entries[0].content == "使用中文回复"
        assert entries[0].type == MemoryType.PREFERENCE
        assert entries[0].scope == MemoryScope.PROJECT
        assert entries[0].explicit is True
        assert entries[0].source_sessions == ["sess_001"]

    def test_extract_user_scope_memory(self) -> None:
        response = (
            '{"memories": [{'
            '"content": "用户偏好中文回复", '
            '"type": "preference", '
            '"scope": "user", '
            '"confidence": 0.9, '
            '"explicit": true'
            "}]}"
        )
        extractor = _make_extractor(response)
        entries = extractor.extract(
            [HumanMessage(content="以后都用中文回复我")], session_id="sess_user"
        )

        assert len(entries) == 1
        assert entries[0].content == "用户偏好中文回复"
        assert entries[0].scope == MemoryScope.USER

    def test_extract_multiple_memories(self) -> None:
        response = (
            '{"memories": ['
            '{"content": "使用中文回复", "type": "preference", '
            '"scope": "project", "confidence": 0.9, "explicit": true},'
            '{"content": "配置在 pyproject.toml", "type": "fact", '
            '"scope": "project", "confidence": 0.8, "explicit": true}'
            "]}"
        )
        extractor = _make_extractor(response)
        entries = extractor.extract(
            [HumanMessage(content="对话内容")], session_id="sess_002"
        )

        assert len(entries) == 2
        assert entries[0].type == MemoryType.PREFERENCE
        assert entries[1].type == MemoryType.FACT

    def test_extract_empty_result(self) -> None:
        extractor = _make_extractor('{"memories": []}')
        entries = extractor.extract(
            [HumanMessage(content="你好")], session_id="sess_003"
        )
        assert entries == []

    def test_extract_invalid_json(self) -> None:
        extractor = _make_extractor("这不是 JSON")
        entries = extractor.extract(
            [HumanMessage(content="你好")], session_id="sess_004"
        )
        assert entries == []

    def test_extract_code_block_json(self) -> None:
        response = (
            '```json\n{"memories": [{'
            '"content": "使用中文回复", "type": "preference", '
            '"confidence": 0.9, "explicit": true'
            "}]}\n```"
        )
        extractor = _make_extractor(response)
        entries = extractor.extract(
            [HumanMessage(content="对话")], session_id="sess_005"
        )
        assert len(entries) == 1
        assert entries[0].content == "使用中文回复"

    def test_filter_low_confidence_non_explicit(self) -> None:
        response = (
            '{"memories": [{'
            '"content": "用户可能喜欢中文", "type": "preference", '
            '"confidence": 0.3, "explicit": false'
            "}]}"
        )
        extractor = _make_extractor(response)
        entries = extractor.extract(
            [HumanMessage(content="对话")], session_id="sess_006"
        )
        assert entries == []

    def test_keep_high_confidence_non_explicit(self) -> None:
        response = (
            '{"memories": [{'
            '"content": "项目使用 pytest", "type": "fact", '
            '"confidence": 0.95, "explicit": false'
            "}]}"
        )
        extractor = _make_extractor(response)
        entries = extractor.extract(
            [HumanMessage(content="对话")], session_id="sess_007"
        )
        assert len(entries) == 1
        assert entries[0].content == "项目使用 pytest"

    def test_missing_llm_returns_empty(self) -> None:
        extractor = MemoryExtractor(llm_factory=None)
        entries = extractor.extract(
            [HumanMessage(content="对话")], session_id="sess_008"
        )
        assert entries == []

    def test_empty_messages_returns_empty(self) -> None:
        extractor = _make_extractor('{"memories": []}')
        entries = extractor.extract([], session_id="sess_009")
        assert entries == []


class TestMemoryExtractorIntegration:
    def test_extract_and_store(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        store = MemoryStore(work_dir=str(isolated_work_dir), root=storage_root)
        response = (
            '{"memories": [{'
            '"content": "使用中文回复", "type": "preference", '
            '"scope": "project", "confidence": 0.9, "explicit": true'
            "}]}"
        )
        extractor = _make_extractor(response)

        messages = [HumanMessage(content="以后都用中文回复我")]
        entries = extractor.extract(messages, session_id="sess_int")
        for entry in entries:
            store.add_or_update(entry)

        stored = store.list()
        assert len(stored) == 1
        assert stored[0].content == "使用中文回复"
