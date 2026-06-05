"""LLM 模块.

封装 LangChain ChatOpenAI 及 provider-specific 子类.
"""

from ai_coding.llm.kimi_chat import KimiChatOpenAI
from ai_coding.llm.lc_llm import create_lc_llm

__all__ = ["KimiChatOpenAI", "create_lc_llm"]
