"""事件系统测试"""

import time
import pytest
from pure_agent_loop.events import Event, EventType


class TestEventType:
    """事件类型枚举测试"""

    def test_all_event_types_exist(self):
        """应该包含所有预定义事件类型"""
        assert EventType.LOOP_START.value == "loop_start"
        assert EventType.THOUGHT.value == "thought"
        assert EventType.ACTION.value == "action"
        assert EventType.OBSERVATION.value == "observation"
        assert EventType.SOFT_LIMIT.value == "soft_limit"
        assert EventType.ERROR.value == "error"
        assert EventType.LOOP_END.value == "loop_end"


class TestEvent:
    """事件对象测试"""

    def test_create_event(self):
        """应该能创建事件实例"""
        event = Event(
            type=EventType.THOUGHT,
            step=1,
            data={"content": "我需要搜索"},
        )
        assert event.type == EventType.THOUGHT
        assert event.step == 1
        assert event.data["content"] == "我需要搜索"
        assert isinstance(event.timestamp, float)

    def test_event_to_dict(self):
        """应该能转换为字典"""
        event = Event(
            type=EventType.ACTION,
            step=2,
            data={"tool": "search", "args": {"query": "python"}},
        )
        d = event.to_dict()
        assert d["type"] == "action"
        assert d["step"] == 2
        assert d["data"]["tool"] == "search"
        assert "timestamp" in d

    def test_event_factory_methods(self):
        """应该有便捷的工厂方法"""
        event = Event.thought(step=1, content="思考中")
        assert event.type == EventType.THOUGHT
        assert event.data["content"] == "思考中"

        event = Event.action(step=2, tool="search", args={"q": "test"})
        assert event.type == EventType.ACTION
        assert event.data["tool"] == "search"
        assert event.data["args"] == {"q": "test"}

        event = Event.observation(step=2, tool="search", result="结果", duration=1.5)
        assert event.type == EventType.OBSERVATION
        assert event.data["tool"] == "search"
        assert event.data["result"] == "结果"
        assert event.data["duration"] == 1.5

        event = Event.error(step=3, error="出错了", fatal=False)
        assert event.type == EventType.ERROR
        assert event.data["error"] == "出错了"
        assert event.data["fatal"] is False

        event = Event.loop_start(task="测试任务")
        assert event.type == EventType.LOOP_START
        assert event.data["task"] == "测试任务"

        event = Event.loop_end(step=5, stop_reason="completed", content="最终结果")
        assert event.type == EventType.LOOP_END
        assert event.data["stop_reason"] == "completed"

        event = Event.soft_limit(step=10, reason="step_limit", prompt="请调整")
        assert event.type == EventType.SOFT_LIMIT
        assert event.data["reason"] == "step_limit"
