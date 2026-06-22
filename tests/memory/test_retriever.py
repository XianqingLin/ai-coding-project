"""MemoryRetriever 单元测试."""

from pathlib import Path

import pytest

from ai_coding.memory.models import MemoryEntry, MemoryScope, MemoryType
from ai_coding.memory.retriever import MemoryRetriever
from ai_coding.memory.store import MemoryStore


@pytest.fixture
def retriever(isolated_work_dir: Path) -> MemoryRetriever:
    """创建使用隔离存储的 MemoryRetriever."""
    storage_root = isolated_work_dir / "memory_data"
    store = MemoryStore(work_dir=str(isolated_work_dir), root=storage_root)
    return MemoryRetriever(store, top_k=3)


class TestMemoryRetriever:
    def test_empty_store_returns_empty_summary(
        self, retriever: MemoryRetriever
    ) -> None:
        assert retriever.get_summary() == ""
        assert retriever.retrieve() == []

    def test_retrieve_sorted_by_confidence(self, retriever: MemoryRetriever) -> None:
        retriever.store.add_or_update(
            MemoryEntry(content="项目使用 pytest 作为测试框架", confidence=0.4)
        )
        retriever.store.add_or_update(
            MemoryEntry(content="Agent 回复必须使用中文", confidence=0.95)
        )
        retriever.store.add_or_update(
            MemoryEntry(content="配置文件统一放在 pyproject.toml", confidence=0.7)
        )

        entries = retriever.retrieve()
        assert [e.content for e in entries] == [
            "Agent 回复必须使用中文",
            "配置文件统一放在 pyproject.toml",
            "项目使用 pytest 作为测试框架",
        ]

    def test_top_k_limit(self, retriever: MemoryRetriever) -> None:
        for i in range(5):
            retriever.store.add_or_update(
                MemoryEntry(content=f"规则 {i}", confidence=0.9 - i * 0.05)
            )

        entries = retriever.retrieve()
        assert len(entries) == 3

    def test_filter_by_scope(self, retriever: MemoryRetriever) -> None:
        retriever.store.add_or_update(
            MemoryEntry(content="项目规则", scope=MemoryScope.PROJECT)
        )
        retriever.store.add_or_update(
            MemoryEntry(content="用户偏好", scope=MemoryScope.USER)
        )

        entries = retriever.retrieve(scope=MemoryScope.USER)
        assert len(entries) == 1
        assert entries[0].content == "用户偏好"

    def test_invalidated_entries_not_retrieved(
        self, retriever: MemoryRetriever
    ) -> None:
        entry = MemoryEntry(content="已否决", confidence=0.95)
        saved = retriever.store.add_or_update(entry)
        retriever.store.invalidate(saved.id)

        assert retriever.retrieve() == []
        assert retriever.get_summary() == ""

    def test_summary_format(self, retriever: MemoryRetriever) -> None:
        retriever.store.add_or_update(
            MemoryEntry(content="使用中文回复", type=MemoryType.PREFERENCE)
        )
        retriever.store.add_or_update(
            MemoryEntry(content="配置在 pyproject.toml", type=MemoryType.FACT)
        )

        summary = retriever.get_summary()
        assert summary.startswith("- ")
        assert "使用中文回复" in summary
        assert "配置在 pyproject.toml" in summary

    def test_low_confidence_filtered(self, retriever: MemoryRetriever) -> None:
        retriever.store.add_or_update(MemoryEntry(content="很弱的推断", confidence=0.1))
        assert retriever.retrieve() == []


class TestMemoryRetrieverCombineSummaries:
    def test_combine_user_and_project_summaries(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        project_store = MemoryStore(
            work_dir=str(isolated_work_dir),
            root=storage_root,
            scope=MemoryScope.PROJECT,
        )
        user_store = MemoryStore(root=storage_root, scope=MemoryScope.USER)

        project_store.add_or_update(
            MemoryEntry(content="项目使用 pytest", scope=MemoryScope.PROJECT)
        )
        user_store.add_or_update(
            MemoryEntry(content="用户偏好中文", scope=MemoryScope.USER)
        )

        summary = MemoryRetriever.combine_summaries(
            {MemoryScope.USER: user_store, MemoryScope.PROJECT: project_store}
        )

        assert "# 用户偏好" in summary
        assert "# 项目记忆" in summary
        assert "用户偏好中文" in summary
        assert "项目使用 pytest" in summary

    def test_combine_omits_empty_sections(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        project_store = MemoryStore(
            work_dir=str(isolated_work_dir),
            root=storage_root,
            scope=MemoryScope.PROJECT,
        )
        user_store = MemoryStore(root=storage_root, scope=MemoryScope.USER)

        project_store.add_or_update(
            MemoryEntry(content="项目使用 pytest", scope=MemoryScope.PROJECT)
        )

        summary = MemoryRetriever.combine_summaries(
            {MemoryScope.USER: user_store, MemoryScope.PROJECT: project_store}
        )

        assert "# 项目记忆" in summary
        assert "# 用户偏好" not in summary
