"""ToolResult 统一结果结构测试."""

from ai_coding.tools.base import ToolResult


class TestToolResult:
    def test_ok_result(self):
        r = ToolResult.ok("done", metadata={"path": "/tmp/x"})
        assert r.success is True
        assert r.data == "done"
        assert r.metadata == {"path": "/tmp/x"}

    def test_fail_result(self):
        r = ToolResult.fail("[错误] 失败", error_code="VALIDATION_ERROR")
        assert r.success is False
        assert r.data == "[错误] 失败"
        assert r.error_code == "VALIDATION_ERROR"

    def test_from_string_treats_error_prefix_as_failure(self):
        assert ToolResult.from_string("[错误] 文件不存在").success is False
        assert ToolResult.from_string("[超时] 命令执行超时").success is False
        assert ToolResult.from_string("[系统] 用户取消").success is False

    def test_from_string_treats_success_prefix_as_success(self):
        assert ToolResult.from_string("[成功] 写入完成").success is True
        assert ToolResult.from_string("echo: hello").success is True
        assert ToolResult.from_string("").success is True
