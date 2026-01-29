"""pure-agent-loop: 轻量级 ReAct 模式 Agentic Loop 框架"""

__version__ = "0.1.0"

from .agent import Agent, AgentResult
from .tool import tool, Tool, ToolRegistry
from .events import Event, EventType
from .renderer import Renderer
from .limits import LoopLimits
from .retry import RetryConfig
from .llm.base import BaseLLMClient
from .llm.openai_client import OpenAIClient
from .llm.types import LLMResponse, ToolCall, TokenUsage
from .errors import (
    PureAgentLoopError,
    ToolExecutionError,
    LLMError,
    LimitExceededError,
)

__all__ = [
    # 核心入口
    "Agent",
    "AgentResult",
    # 工具
    "tool",
    "Tool",
    "ToolRegistry",
    # 事件
    "Event",
    "EventType",
    # 渲染
    "Renderer",
    # 配置
    "LoopLimits",
    "RetryConfig",
    # LLM
    "BaseLLMClient",
    "OpenAIClient",
    "LLMResponse",
    "ToolCall",
    "TokenUsage",
    # 异常
    "PureAgentLoopError",
    "ToolExecutionError",
    "LLMError",
    "LimitExceededError",
]
