"""LangChain LLM 工厂模块.

根据提供商名称创建对应的 LangChain ChatOpenAI 实例.
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
from ai_coding.llm.kimi_chat import KimiChatOpenAI
from ai_coding.logger import get_logger

logger = get_logger(__name__)


def create_lc_llm(provider: str = "kimi") -> ChatOpenAI:
    """根据提供商名称创建 LangChain ChatOpenAI 实例.

    Args:
        provider: LLM 提供商名称. 可选: "kimi", "openai", "mock".

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

    elif provider == "mock":
        from ai_coding.mock_llm import MockChatModel

        logger.info("创建 Mock LLM")
        return MockChatModel()

    else:
        raise ValueError(
            f"未知的 LLM 提供商: '{provider}'. 可选: kimi, openai, mock"
        )
