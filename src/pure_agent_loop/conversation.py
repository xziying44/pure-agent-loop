"""多轮对话会话

提供 Conversation 类，自动维护消息历史，支持多轮对话续接。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from .events import Event, EventType

if TYPE_CHECKING:
    from .agent import Agent, AgentResult


class Conversation:
    """多轮对话会话

    由 Agent.conversation() 创建，自动维护消息历史。
    同一 Agent 可创建多个独立 Conversation。

    使用示例:
        agent = Agent(llm=my_llm)
        conv = agent.conversation()

        # 多轮对话自动续接
        async for event in conv.send_stream("第一个问题"):
            ...
        async for event in conv.send_stream("追问"):
            ...
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """当前消息历史（只读副本）"""
        return list(self._messages)

    def reset(self) -> None:
        """清空消息历史，开始新对话"""
        self._messages = []
