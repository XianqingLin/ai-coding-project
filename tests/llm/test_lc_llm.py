"""LangChain LLM 工厂测试."""

import pytest

from ai_coding.llm.lc_llm import create_lc_llm
from ai_coding.mock_llm import MockChatModel


class TestCreateLCLLM:
    def test_create_mock_llm(self):
        llm = create_lc_llm("mock")
        assert isinstance(llm, MockChatModel)

    def test_provider_case_insensitive(self):
        llm = create_lc_llm("MOCK")
        assert isinstance(llm, MockChatModel)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="未知的 LLM 提供商"):
            create_lc_llm("unknown")

    def test_kimi_without_key_raises(self, monkeypatch):
        monkeypatch.setattr("ai_coding.llm.lc_llm.KIMI_API_KEY", "")
        with pytest.raises(ValueError, match="Kimi API Key 未设置"):
            create_lc_llm("kimi")

    def test_kimi_with_placeholder_key_raises(self, monkeypatch):
        monkeypatch.setattr("ai_coding.llm.lc_llm.KIMI_API_KEY", "your_kimi_api_key_here")
        with pytest.raises(ValueError, match="Kimi API Key 未设置"):
            create_lc_llm("kimi")

    def test_openai_without_key_raises(self, monkeypatch):
        monkeypatch.setattr("ai_coding.llm.lc_llm.OPENAI_API_KEY", "")
        with pytest.raises(ValueError, match="OpenAI API Key 未设置"):
            create_lc_llm("openai")
