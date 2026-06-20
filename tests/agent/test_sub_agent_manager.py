"""子 Agent 管理器测试."""

from ai_coding.agent.sub_agent_manager import SubAgentManager
from ai_coding.mock_llm import MockChatModel, mock_text


class TestSubAgentManagerSingleton:
    """单例行为测试."""

    def test_singleton_instance(self):
        """SubAgentManager 应为单例."""
        SubAgentManager._instance = None
        m1 = SubAgentManager()
        m2 = SubAgentManager()
        assert m1 is m2
        SubAgentManager._instance = None


class TestSubAgentManagerDispatch:
    """子 Agent 派发测试."""

    def test_sync_dispatch_returns_result(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """同步派发应返回子 Agent 执行结果."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        result = manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        assert "done" in result

    def test_background_dispatch_returns_task_id(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """后台派发应返回任务 ID."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        result = manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
            run_in_background=True,
        )

        assert "后台启动" in result

    def test_background_instance_stored_and_runnable(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """后台派发的实例应被记录，且最终完成."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("background result")])

        result = manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
            run_in_background=True,
        )

        # 提取 instance_id
        import re

        match = re.search(r"ID: (sub_[a-f0-9]+)", result)
        assert match
        sid = match.group(1)

        instance = manager.get_instance(sid)
        assert instance is not None
        assert instance.status in ("running", "completed", "failed")

        # 等待完成
        import time

        for _ in range(50):
            instance = manager.get_instance(sid)
            if instance.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert instance.status == "completed"
        assert "background result" in instance.result


class TestSubAgentManagerToolsAndPrompts:
    """工具和 Prompt 配置测试."""

    def test_explore_agent_limited_tools(self):
        """explore 类型子 Agent 应只拥有只读工具."""
        manager = SubAgentManager()
        tools = manager._get_tools_for_type("explore")
        tool_names = {t.name for t in tools}

        assert tool_names == {"read_file", "list_dir", "grep", "glob"}

    def test_coder_agent_has_all_tools(self):
        """coder 类型子 Agent 应拥有全部工具."""
        manager = SubAgentManager()
        tools = manager._get_tools_for_type("coder")
        tool_names = {t.name for t in tools}

        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "execute_command" in tool_names

    def test_unknown_agent_type_uses_all_tools(self):
        """未知类型回退到使用全部工具."""
        manager = SubAgentManager()
        tools = manager._get_tools_for_type("unknown")
        tool_names = {t.name for t in tools}
        assert "write_file" in tool_names

    def test_explore_prompt_is_readonly(self):
        """explore 类型应使用只读专用 prompt."""
        manager = SubAgentManager()
        prompt = manager._get_prompt_for_type("explore")

        assert prompt is not None
        assert "read_file" in prompt
        assert "不要尝试写入" in prompt

    def test_coder_prompt_is_default(self):
        """coder 类型 prompt 应为默认（None）."""
        manager = SubAgentManager()
        prompt = manager._get_prompt_for_type("coder")

        assert prompt is None

    def test_unknown_prompt_is_none(self):
        """未知类型 prompt 回退到 None."""
        manager = SubAgentManager()
        assert manager._get_prompt_for_type("unknown") is None


class TestSubAgentManagerQuery:
    """查询和通知状态测试."""

    def test_list_instances_by_status(self, clean_sub_agent_manager, isolated_work_dir):
        """list_instances 应能按状态过滤."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        completed = manager.list_instances(status="completed")
        running = manager.list_instances(status="running")

        assert len(completed) >= 1
        assert len(running) == 0

    def test_pending_notifications_and_mark_notified(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """已完成实例应出现在 pending notifications，标记后消失."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        pending = manager.get_pending_notifications()
        assert len(pending) >= 1
        sid = pending[0].instance_id

        manager.mark_notified(sid)
        pending = manager.get_pending_notifications()
        assert all(i.instance_id != sid for i in pending)

    def test_to_dict(self, clean_sub_agent_manager, isolated_work_dir):
        """to_dict 应正确序列化实例字段."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        manager.dispatch(
            agent_type="coder",
            prompt="say hello",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        instance = manager.list_instances()[0]
        data = manager.to_dict(instance)

        assert data["instance_id"] == instance.instance_id
        assert data["agent_type"] == "coder"
        assert data["status"] == "completed"
        assert "task" in data


class TestSubAgentManagerFailure:
    """失败场景测试."""

    def test_sync_dispatch_failure(self, clean_sub_agent_manager, isolated_work_dir):
        """子 Agent 执行异常时，结果应包含错误信息."""
        manager = clean_sub_agent_manager

        class FailingLLM(MockChatModel):
            def invoke(self, messages, **kwargs):
                raise RuntimeError("mock llm failure")

        result = manager.dispatch(
            agent_type="coder",
            prompt="trigger failure",
            llm=FailingLLM(),
            work_dir=str(isolated_work_dir),
        )

        # LangGraphAgent.run() 内部已捕获异常并返回错误字符串，
        # 所以子 Agent 结果中应包含错误提示。
        assert "Agent 执行失败" in result or "错误" in result

        instances = manager.list_instances()
        assert len(instances) == 1
        assert "mock llm failure" in instances[0].result

    def test_get_nonexistent_instance(self, clean_sub_agent_manager):
        """获取不存在的实例应返回 None."""
        manager = clean_sub_agent_manager
        assert manager.get_instance("not_exist") is None

    def test_resume_running_instance(self, clean_sub_agent_manager, isolated_work_dir):
        """唤回仍在运行中的实例应返回错误信息."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        # 后台启动一个长时间运行的子 Agent
        result = manager.dispatch(
            agent_type="coder",
            prompt="long task",
            llm=llm,
            work_dir=str(isolated_work_dir),
            run_in_background=True,
        )
        sid = result.split("ID: ")[1].split("\n")[0].strip()

        # 在运行期间尝试唤回
        resume_result = manager.dispatch(
            agent_type="coder",
            prompt="resume",
            llm=llm,
            instance_id=sid,
        )
        assert "仍在运行" in resume_result


class TestSubAgentManagerResume:
    """唤回（resume）实例测试."""

    def test_resume_completed_instance_sync(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """同步唤回已完成实例并执行新任务."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        manager.dispatch(
            agent_type="coder",
            prompt="first",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )
        sid = manager.list_instances()[0].instance_id

        # 使用全新的 LLM 唤回，避免响应耗尽
        resume_llm = MockChatModel(responses=[mock_text("resumed result")])
        result = manager.dispatch(
            agent_type="coder",
            prompt="second",
            llm=resume_llm,
            instance_id=sid,
            work_dir=str(isolated_work_dir),
        )
        assert "resumed result" in result
        assert manager.get_instance(sid).status == "completed"

    def test_resume_completed_instance_background(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """后台唤回已完成实例."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("done")])

        manager.dispatch(
            agent_type="coder",
            prompt="first",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )
        sid = manager.list_instances()[0].instance_id

        resume_llm = MockChatModel(responses=[mock_text("bg resumed")])
        result = manager.dispatch(
            agent_type="coder",
            prompt="second",
            llm=resume_llm,
            instance_id=sid,
            work_dir=str(isolated_work_dir),
            run_in_background=True,
        )
        assert "唤回并后台运行" in result
        assert sid in result

    def test_resume_nonexistent_instance(self, clean_sub_agent_manager):
        """直接唤回不存在的实例返回错误."""
        manager = clean_sub_agent_manager
        result = manager._resume_instance(
            instance_id="not_exist",
            prompt="x",
            llm=MockChatModel(),
            llm_factory=None,
            run_in_background=False,
        )
        assert "实例不存在" in result

    def test_resume_with_llm_factory(self, clean_sub_agent_manager, isolated_work_dir):
        """唤回时通过 llm_factory 提供新 LLM."""
        manager = clean_sub_agent_manager
        llm = MockChatModel(responses=[mock_text("first")])

        manager.dispatch(
            agent_type="coder",
            prompt="first",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )
        sid = manager.list_instances()[0].instance_id

        new_llm = MockChatModel(responses=[mock_text("factory result")])
        result = manager.dispatch(
            agent_type="coder",
            prompt="second",
            llm=None,
            llm_factory=lambda: new_llm,
            instance_id=sid,
            work_dir=str(isolated_work_dir),
        )
        assert "factory result" in result
