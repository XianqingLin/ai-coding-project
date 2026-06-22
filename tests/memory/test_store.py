"""MemoryStore 单元测试."""

from pathlib import Path

import pytest

from ai_coding.memory.models import MemoryEntry, MemoryScope, MemoryType
from ai_coding.memory.store import MemoryStore, _content_similarity


@pytest.fixture
def memory_store(isolated_work_dir: Path) -> MemoryStore:
    """创建使用隔离存储根目录的 MemoryStore."""
    storage_root = isolated_work_dir / "memory_data"
    return MemoryStore(work_dir=str(isolated_work_dir), root=storage_root)


class TestContentSimilarity:
    def test_identical_text(self) -> None:
        assert _content_similarity("使用中文回复", "使用中文回复") == 1.0

    def test_substring_match(self) -> None:
        score = _content_similarity("使用中文", "请使用中文回复我")
        assert score == 0.9

    def test_jaccard_overlap(self) -> None:
        score = _content_similarity("使用中文回复", "使用英文回复")
        assert 0.0 < score < 1.0

    def test_no_overlap(self) -> None:
        score = _content_similarity("abcdef", "xyz123")
        assert score == 0.0


class TestMemoryStoreAddAndList:
    def test_add_single_entry(self, memory_store: MemoryStore) -> None:
        entry = MemoryEntry(
            content="使用中文回复",
            type=MemoryType.PREFERENCE,
            scope=MemoryScope.PROJECT,
            explicit=True,
            confidence=0.9,
            source_sessions=["sess_001"],
        )
        saved = memory_store.add_or_update(entry)

        assert saved.id == entry.id
        entries = memory_store.list()
        assert len(entries) == 1
        assert entries[0].content == "使用中文回复"
        assert entries[0].type == MemoryType.PREFERENCE

    def test_similar_entries_merge(self, memory_store: MemoryStore) -> None:
        entry1 = MemoryEntry(
            content="使用中文回复",
            type=MemoryType.PREFERENCE,
            explicit=True,
            confidence=0.8,
            source_sessions=["sess_001"],
        )
        entry2 = MemoryEntry(
            content="请使用中文回复我",
            type=MemoryType.PREFERENCE,
            explicit=True,
            confidence=0.9,
            source_sessions=["sess_002"],
        )
        memory_store.add_or_update(entry1)
        saved = memory_store.add_or_update(entry2)

        entries = memory_store.list()
        assert len(entries) == 1
        assert entries[0].observation_count == 2
        assert set(entries[0].source_sessions) == {"sess_001", "sess_002"}
        assert entries[0].confidence == 0.9
        assert saved.id == entry1.id

    def test_different_entries_do_not_merge(self, memory_store: MemoryStore) -> None:
        entry1 = MemoryEntry(content="使用中文回复")
        entry2 = MemoryEntry(content="项目使用 pytest 做单元测试")
        memory_store.add_or_update(entry1)
        memory_store.add_or_update(entry2)

        entries = memory_store.list()
        assert len(entries) == 2

    def test_filter_by_scope(self, memory_store: MemoryStore) -> None:
        project_entry = MemoryEntry(
            content="项目使用 pytest",
            scope=MemoryScope.PROJECT,
        )
        user_entry = MemoryEntry(
            content="用户偏好中文回复",
            scope=MemoryScope.USER,
        )
        memory_store.add_or_update(project_entry)
        memory_store.add_or_update(user_entry)

        project_entries = memory_store.list(scope=MemoryScope.PROJECT)
        assert len(project_entries) == 1
        assert project_entries[0].scope == MemoryScope.PROJECT


class TestMemoryStoreLifecycle:
    def test_get_existing_entry(self, memory_store: MemoryStore) -> None:
        entry = MemoryEntry(content="配置在 pyproject.toml")
        saved = memory_store.add_or_update(entry)

        found = memory_store.get(saved.id)
        assert found is not None
        assert found.content == "配置在 pyproject.toml"

    def test_get_missing_entry(self, memory_store: MemoryStore) -> None:
        assert memory_store.get("not-exist") is None

    def test_delete_entry(self, memory_store: MemoryStore) -> None:
        entry = MemoryEntry(content="待删除")
        saved = memory_store.add_or_update(entry)

        assert memory_store.delete(saved.id) is True
        assert memory_store.list() == []
        assert memory_store.delete(saved.id) is False

    def test_invalidate_entry(self, memory_store: MemoryStore) -> None:
        entry = MemoryEntry(content="已否决的规则")
        saved = memory_store.add_or_update(entry)

        assert memory_store.invalidate(saved.id) is True
        entries = memory_store.list(include_invalidated=False)
        assert len(entries) == 0

        all_entries = memory_store.list(include_invalidated=True)
        assert len(all_entries) == 1
        assert all_entries[0].invalidated is True

    def test_invalidate_missing_entry(self, memory_store: MemoryStore) -> None:
        assert memory_store.invalidate("not-exist") is False


class TestMemoryStorePersistence:
    def test_data_survives_recreate(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        store1 = MemoryStore(work_dir=str(isolated_work_dir), root=storage_root)
        entry = MemoryEntry(content="持久化测试")
        store1.add_or_update(entry)

        store2 = MemoryStore(work_dir=str(isolated_work_dir), root=storage_root)
        entries = store2.list()
        assert len(entries) == 1
        assert entries[0].content == "持久化测试"

    def test_different_work_dirs_are_isolated(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        dir_a = isolated_work_dir / "project_a"
        dir_b = isolated_work_dir / "project_b"

        store_a = MemoryStore(work_dir=str(dir_a), root=storage_root)
        store_a.add_or_update(MemoryEntry(content="项目 A 的规则"))

        store_b = MemoryStore(work_dir=str(dir_b), root=storage_root)
        store_b.add_or_update(MemoryEntry(content="项目 B 的规则"))

        assert len(store_a.list()) == 1
        assert len(store_b.list()) == 1
        assert store_a.list()[0].content != store_b.list()[0].content


class TestMemoryStoreUserScope:
    def test_user_scope_uses_global_path(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"
        store = MemoryStore(root=storage_root, scope=MemoryScope.USER)
        store.add_or_update(
            MemoryEntry(content="用户偏好中文回复", scope=MemoryScope.USER)
        )

        assert (storage_root / "memory" / "global" / "memory.jsonl").exists()
        entries = store.list()
        assert len(entries) == 1
        assert entries[0].content == "用户偏好中文回复"

    def test_user_scope_is_cross_project(self, isolated_work_dir: Path) -> None:
        storage_root = isolated_work_dir / "memory_data"

        store_a = MemoryStore(root=storage_root, scope=MemoryScope.USER)
        store_a.add_or_update(MemoryEntry(content="用户偏好", scope=MemoryScope.USER))

        store_b = MemoryStore(root=storage_root, scope=MemoryScope.USER)
        entries = store_b.list()
        assert len(entries) == 1
        assert entries[0].content == "用户偏好"

    def test_project_scope_requires_work_dir(self) -> None:
        with pytest.raises(ValueError):
            MemoryStore(work_dir="", scope=MemoryScope.PROJECT)
