"""事件系统

定义 Agentic Loop 执行过程中产出的结构化事件。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """事件类型枚举"""

    LOOP_START = "loop_start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SOFT_LIMIT = "soft_limit"
    ERROR = "error"
    LOOP_END = "loop_end"
    TODO_UPDATE = "todo_update"


@dataclass
class Event:
    """结构化事件

    Attributes:
        type: 事件类型
        step: 当前步数
        data: 事件数据
        timestamp: 事件产生时间戳
    """

    type: EventType
    step: int
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典"""
        return {
            "type": self.type.value,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    # ---- 工厂方法 ----

    @classmethod
    def thought(cls, step: int, content: str) -> "Event":
        """创建思考事件"""
        return cls(type=EventType.THOUGHT, step=step, data={"content": content})

    @classmethod
    def action(cls, step: int, tool: str, args: dict[str, Any]) -> "Event":
        """创建工具调用事件"""
        return cls(type=EventType.ACTION, step=step, data={"tool": tool, "args": args})

    @classmethod
    def observation(
        cls, step: int, tool: str, result: str, duration: float
    ) -> "Event":
        """创建观察结果事件"""
        return cls(
            type=EventType.OBSERVATION,
            step=step,
            data={"tool": tool, "result": result, "duration": duration},
        )

    @classmethod
    def error(cls, step: int, error: str, fatal: bool = False) -> "Event":
        """创建错误事件"""
        return cls(
            type=EventType.ERROR,
            step=step,
            data={"error": error, "fatal": fatal},
        )

    @classmethod
    def loop_start(cls, task: str) -> "Event":
        """创建循环启动事件"""
        return cls(type=EventType.LOOP_START, step=0, data={"task": task})

    @classmethod
    def loop_end(
        cls, step: int, stop_reason: str, content: str = "", messages: list | None = None
    ) -> "Event":
        """创建循环结束事件"""
        return cls(
            type=EventType.LOOP_END,
            step=step,
            data={"stop_reason": stop_reason, "content": content, "messages": messages or []},
        )

    @classmethod
    def soft_limit(cls, step: int, reason: str, prompt: str) -> "Event":
        """创建软限制事件"""
        return cls(
            type=EventType.SOFT_LIMIT,
            step=step,
            data={"reason": reason, "prompt": prompt},
        )

    @classmethod
    def todo_update(cls, step: int, todos: list[dict]) -> "Event":
        """创建任务列表变更事件"""
        return cls(
            type=EventType.TODO_UPDATE,
            step=step,
            data={"todos": todos},
        )
