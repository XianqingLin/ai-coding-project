"""Kimi ChatOpenAI 子类.

提取 API 响应中的 reasoning_content 字段.
LangChain 的 ChatOpenAI 不自动解析此字段, 需手动拦截.
"""

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from ai_coding.logger import get_logger

logger = get_logger(__name__)


class KimiChatOpenAI(ChatOpenAI):
    """支持 Kimi K2.6 reasoning_content 的 ChatOpenAI 子类.

    功能:
    1. 请求端: 为包含 tool_calls 的 assistant 消息注入空的 reasoning_content,
       避免 Kimi API 报 400 错误.
    2. 响应端(非流式): 重载 _create_chat_result, 从原始 API 响应中提取
       reasoning_content 注入 AIMessage.additional_kwargs.
    3. 响应端(流式): 重载 _convert_chunk_to_generation_chunk, 从每个 streaming
       chunk 的 delta 中提取 reasoning_content, 利用 LangChain 的 chunk 合并
       机制自动拼接到最终的 AIMessage.additional_kwargs.

    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """重写请求 payload 构建, 注入 reasoning_content."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        for message in payload.get("messages", []):
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and "reasoning_content" not in message
            ):
                message["reasoning_content"] = ""

        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """重写非流式响应解析, 提取 reasoning_content.

        调用父类方法完成标准解析后, 从原始 OpenAI SDK 响应对象中
        读取 message.reasoning_content, 注入到 AIMessage.additional_kwargs.

        """
        result = super()._create_chat_result(response, generation_info)

        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            for i, choice in enumerate(response.choices):
                if i >= len(result.generations):
                    break
                msg = choice.message
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    result.generations[i].message.additional_kwargs[
                        "reasoning_content"
                    ] = msg.reasoning_content
                    logger.debug(
                        f"提取 reasoning_content: {len(msg.reasoning_content)} 字符"
                    )

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """重写流式 chunk 解析, 提取 reasoning_content.

        LangChain 的 _stream 路径会调用此方法把每个原始 API chunk 转换为
        ChatGenerationChunk. 父类实现不识别 reasoning_content, 导致流式输出
        下该字段丢失.

        这里先调用父类完成标准转换, 然后从原始 chunk 的 delta 中读取
        reasoning_content 并写入 message.additional_kwargs.

        由于 LangChain 的 invoke() 在 streaming=True 时会收集所有 chunk 并通过
        AIMessageChunk.__add__ 合并, 其中 additional_kwargs 会按字符串拼接策略
        累加, 因此分散在多 chunk 中的 reasoning_content 最终会合并为完整内容.

        """
        result = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if result is None:
            return None

        # 从原始 chunk dict 中提取 delta.reasoning_content
        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content:
                if isinstance(result.message, AIMessageChunk):
                    result.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                    logger.debug(
                        f"流式提取 reasoning_content chunk: {len(reasoning_content)} 字符"
                    )

        return result
