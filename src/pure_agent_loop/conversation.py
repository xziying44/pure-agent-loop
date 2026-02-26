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

    async def send_stream(self, task: str) -> AsyncIterator[Event]:
        """发送消息并流式返回事件

        自动将内部维护的消息历史传递给 Agent，
        并在 LOOP_END 事件中捕获更新后的消息历史。

        Args:
            task: 任务描述

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 首次调用时 messages 为空 → 传 None（等同于新对话）
        # 后续调用时 messages 非空 → 传入历史实现续接
        msgs = self._messages if self._messages else None

        async for event in self._agent.arun_stream(task, messages=msgs):
            # 捕获 LOOP_END 事件，更新内部消息历史
            if event.type == EventType.LOOP_END:
                self._messages = event.data.get("messages", [])
            yield event
