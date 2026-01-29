"""LLM 相关类型定义

定义 LLM 调用的统一响应模型，屏蔽不同提供商的差异。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token 用量统计

    Attributes:
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        total_tokens: 总 token 数
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """累加两个 TokenUsage"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """创建零值 TokenUsage"""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@dataclass
class ToolCall:
    """工具调用信息

    Attributes:
        id: 调用 ID（用于匹配 tool 消息）
        name: 工具名称
        arguments: 解析后的参数字典
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 统一响应模型

    Attributes:
        content: 文本回复内容（可能为 None）
        tool_calls: 工具调用列表
        usage: Token 用量统计
        raw: 原始响应对象（供高级用户访问）
    """

    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    raw: Any

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0
