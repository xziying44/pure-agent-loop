"""LLM 客户端抽象接口

定义所有 LLM 客户端必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from .types import LLMResponse


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类

    所有 LLM 客户端（内置和自定义）必须继承此类并实现 chat 方法。
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话请求

        Args:
            messages: 消息列表（OpenAI Chat 格式）
            tools: 工具定义列表（OpenAI Function Calling 格式）
            **kwargs: 额外参数（如 temperature 等）

        Returns:
            统一响应模型
        """
        ...
