"""LangChain LLM 封装模块.

使用 langchain_openai.ChatOpenAI 封装 Kimi / OpenAI 模型.
Kimi 兼容 OpenAI API 格式，可直接使用 ChatOpenAI.

针对 Kimi K2.6 的 thinking 模式做了特殊处理：
- Kimi 在 tool_calls 场景中要求请求中包含 reasoning_content
- LangChain 的 ChatOpenAI 默认不传递此字段
- 通过子类化 _get_request_payload 自动注入
"""

from langchain_openai import ChatOpenAI

from ai_coding.config import (
    KIMI_API_KEY,
    KIMI_BASE_URL,
    KIMI_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from ai_coding.logger import get_logger

logger = get_logger(__name__)


class KimiChatOpenAI(ChatOpenAI):
    """支持 Kimi K2.6 reasoning_content 的 ChatOpenAI 子类.

    Kimi K2.6 在 assistant message 包含 tool_calls 时，
    要求请求中同时包含 reasoning_content 字段.
    LangChain 默认不传递此字段，导致 400 错误.

    此类在构建 API 请求 payload 时，自动为包含 tool_calls 的
    assistant 消息注入空的 reasoning_content.

    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """重写请求 payload 构建，注入 reasoning_content."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 为包含 tool_calls 的 assistant 消息注入 reasoning_content
        for message in payload.get("messages", []):
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and "reasoning_content" not in message
            ):
                message["reasoning_content"] = ""

        return payload


def create_lc_llm(provider: str = "kimi") -> ChatOpenAI:
    """根据提供商名称创建 LangChain ChatOpenAI 实例.

    Args:
        provider: LLM 提供商名称. 可选: "kimi", "openai".

    Returns:
        ChatOpenAI 实例.

    Raises:
        ValueError: 未知的提供商或 API Key 未设置.

    """
    provider = provider.lower()

    if provider == "kimi":
        if not KIMI_API_KEY or KIMI_API_KEY == "your_kimi_api_key_here":
            raise ValueError(
                "Kimi API Key 未设置. 请在 .env 文件中设置 KIMI_API_KEY."
            )

        logger.info(f"创建 Kimi LLM | 模型: {KIMI_MODEL} | URL: {KIMI_BASE_URL}")
        return KimiChatOpenAI(
            model=KIMI_MODEL,
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
            temperature=1.0,
            streaming=True,
        )

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API Key 未设置. 请在 .env 文件中设置 OPENAI_API_KEY."
            )

        logger.info(f"创建 OpenAI LLM | 模型: {OPENAI_MODEL}")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.7,
            streaming=True,
        )

    else:
        raise ValueError(
            f"未知的 LLM 提供商: '{provider}'. 可选: kimi, openai"
        )
