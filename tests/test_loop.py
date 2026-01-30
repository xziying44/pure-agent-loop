"""ReAct 循环引擎测试"""

import json
import pytest
from unittest.mock import AsyncMock

from pure_agent_loop.loop import ReactLoop
from pure_agent_loop.tool import tool, ToolRegistry
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage
from pure_agent_loop.events import Event, EventType
from pure_agent_loop.limits import LoopLimits
from pure_agent_loop.retry import RetryConfig


class MockLLM(BaseLLMClient):
    """可控的 Mock LLM 客户端"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _text_response(content: str) -> LLMResponse:
    """创建纯文本响应"""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        raw={},
    )


def _tool_call_response(tool_name: str, arguments: dict) -> LLMResponse:
    """创建工具调用响应"""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call_{tool_name}", name=tool_name, arguments=arguments)],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )


def _multi_tool_call_response(calls: list[tuple[str, dict]]) -> LLMResponse:
    """创建多工具调用响应

    Args:
        calls: [(tool_name, arguments), ...] 格式的调用列表
    """
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id=f"call_{name}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )


class TestReactLoop:
    """ReactLoop 测试"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """无工具调用时应直接返回文本"""
        llm = MockLLM([_text_response("你好，世界！")])

        loop = ReactLoop(
            llm=llm,
            tool_registry=ToolRegistry(),
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("打个招呼", system_prompt="你是助手"):
            events.append(event)

        # 应该有: LOOP_START, THOUGHT, LOOP_END
        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

        # 最终结果
        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        assert end_event.data["stop_reason"] == "completed"
        assert end_event.data["content"] == "你好，世界！"

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        """应该执行工具调用然后返回最终结果"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        registry = ToolRegistry()
        registry.register(add)

        llm = MockLLM([
            _tool_call_response("add", {"a": 3, "b": 4}),
            _text_response("3 + 4 = 7"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("计算 3+4"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.ACTION in types
        assert EventType.OBSERVATION in types
        assert EventType.LOOP_END in types

        # 检查工具执行结果
        obs_event = next(e for e in events if e.type == EventType.OBSERVATION)
        assert obs_event.data["result"] == "7"

    @pytest.mark.asyncio
    async def test_step_soft_limit_triggers(self):
        """步数软限制应触发并注入提示"""

        @tool
        def noop(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(noop)

        # 创建 4 次工具调用 + 最终文本响应
        responses = []
        for i in range(4):
            responses.append(_tool_call_response("noop", {"x": str(i)}))
        responses.append(_text_response("完成"))

        llm = MockLLM(responses)

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(max_steps=3),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试软限制"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.SOFT_LIMIT in types

    @pytest.mark.asyncio
    async def test_token_hard_limit_stops(self):
        """token 硬限制应强制终止循环"""

        @tool
        def noop(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(noop)

        # 创建多次工具调用
        responses = [_tool_call_response("noop", {"x": str(i)}) for i in range(20)]
        llm = MockLLM(responses)

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(max_tokens=200),  # 非常低的限制
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试硬限制"):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        assert end_event.data["stop_reason"] == "token_limit"

    @pytest.mark.asyncio
    async def test_messages_history_continuity(self):
        """应该返回完整的消息历史用于多轮对话"""
        llm = MockLLM([_text_response("你好")])

        loop = ReactLoop(
            llm=llm,
            tool_registry=ToolRegistry(),
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("hi", system_prompt="你是助手"):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        messages = end_event.data.get("messages", [])
        assert len(messages) >= 2  # 至少有 system + user + assistant

    @pytest.mark.asyncio
    async def test_todo_write_emits_todo_update_event(self):
        """执行 todo_write 工具应额外产出 TODO_UPDATE 事件"""
        from pure_agent_loop.builtin_tools import TodoStore, create_todo_tool

        store = TodoStore()
        todo_tool = create_todo_tool(store)

        registry = ToolRegistry()
        registry.register(todo_tool)

        llm = MockLLM([
            _tool_call_response("todo_write", {
                "todos": [
                    {"content": "搜索资料", "status": "completed"},
                    {"content": "分析结果", "status": "pending"},
                ]
            }),
            _text_response("已规划任务"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
            todo_store=store,
        )

        events = []
        async for event in loop.run("测试任务"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.TODO_UPDATE in types

        todo_event = next(e for e in events if e.type == EventType.TODO_UPDATE)
        assert len(todo_event.data["todos"]) == 2
        assert todo_event.data["todos"][0]["content"] == "搜索资料"

    @pytest.mark.asyncio
    async def test_no_todo_update_without_store(self):
        """未传入 todo_store 时不应产出 TODO_UPDATE 事件"""

        @tool
        def dummy(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(dummy)

        llm = MockLLM([
            _tool_call_response("dummy", {"x": "test"}),
            _text_response("完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.TODO_UPDATE not in types

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_event_order(self):
        """并行执行时应先产出所有 ACTION，再产出所有 OBSERVATION"""
        import asyncio

        @tool
        async def slow_tool(name: str) -> str:
            """模拟耗时工具"""
            await asyncio.sleep(0.05)
            return f"result_{name}"

        registry = ToolRegistry()
        registry.register(slow_tool)

        llm = MockLLM([
            _multi_tool_call_response([
                ("slow_tool", {"name": "a"}),
                ("slow_tool", {"name": "b"}),
                ("slow_tool", {"name": "c"}),
            ]),
            _text_response("完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试并行"):
            events.append(event)

        # 提取 ACTION 和 OBSERVATION 事件的索引
        action_indices = [i for i, e in enumerate(events) if e.type == EventType.ACTION]
        obs_indices = [i for i, e in enumerate(events) if e.type == EventType.OBSERVATION]

        # 应该有 3 个 ACTION 和 3 个 OBSERVATION
        assert len(action_indices) == 3, f"期望 3 个 ACTION，实际 {len(action_indices)}"
        assert len(obs_indices) == 3, f"期望 3 个 OBSERVATION，实际 {len(obs_indices)}"

        # 所有 ACTION 应该在所有 OBSERVATION 之前
        assert max(action_indices) < min(obs_indices), \
            "所有 ACTION 事件应该在所有 OBSERVATION 事件之前"

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self):
        """并行执行应该节省时间：3 个 0.1 秒工具应在 0.2 秒内完成"""
        import asyncio
        import time

        @tool
        async def timed_tool(name: str) -> str:
            """耗时 0.1 秒的工具"""
            await asyncio.sleep(0.1)
            return f"done_{name}"

        registry = ToolRegistry()
        registry.register(timed_tool)

        llm = MockLLM([
            _multi_tool_call_response([
                ("timed_tool", {"name": "a"}),
                ("timed_tool", {"name": "b"}),
                ("timed_tool", {"name": "c"}),
            ]),
            _text_response("完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        start = time.time()
        events = []
        async for event in loop.run("测试性能"):
            events.append(event)
        elapsed = time.time() - start

        # 串行需要 0.3 秒，并行应该 < 0.2 秒
        assert elapsed < 0.2, f"并行执行耗时 {elapsed:.3f}s，应该 < 0.2s"

    @pytest.mark.asyncio
    async def test_parallel_execution_partial_failure(self):
        """部分工具失败不应影响其他工具的执行"""

        @tool
        def success_tool(x: str) -> str:
            """成功的工具"""
            return f"success_{x}"

        @tool
        def fail_tool(x: str) -> str:
            """失败的工具"""
            raise ValueError("故意失败")

        registry = ToolRegistry()
        registry.register(success_tool)
        registry.register(fail_tool)

        llm = MockLLM([
            _multi_tool_call_response([
                ("success_tool", {"x": "a"}),
                ("fail_tool", {"x": "b"}),
                ("success_tool", {"x": "c"}),
            ]),
            _text_response("处理完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试部分失败"):
            events.append(event)

        # 应该有 3 个 OBSERVATION
        obs_events = [e for e in events if e.type == EventType.OBSERVATION]
        assert len(obs_events) == 3, f"期望 3 个 OBSERVATION，实际 {len(obs_events)}"

        # 检查结果：2 个成功，1 个包含错误信息
        results = [e.data["result"] for e in obs_events]
        success_count = sum(1 for r in results if r.startswith("success_"))
        error_count = sum(1 for r in results if "执行失败" in r)

        assert success_count == 2, f"期望 2 个成功，实际 {success_count}"
        assert error_count == 1, f"期望 1 个错误，实际 {error_count}"

        # 循环应该正常结束
        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        assert end_event.data["stop_reason"] == "completed"

    @pytest.mark.asyncio
    async def test_parallel_execution_individual_timing(self):
        """每个工具应该有独立的执行时间记录"""
        import asyncio

        @tool
        async def fast_tool(x: str) -> str:
            """快速工具 0.05 秒"""
            await asyncio.sleep(0.05)
            return "fast"

        @tool
        async def slow_tool(x: str) -> str:
            """慢速工具 0.15 秒"""
            await asyncio.sleep(0.15)
            return "slow"

        registry = ToolRegistry()
        registry.register(fast_tool)
        registry.register(slow_tool)

        llm = MockLLM([
            _multi_tool_call_response([
                ("fast_tool", {"x": "a"}),
                ("slow_tool", {"x": "b"}),
            ]),
            _text_response("完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试计时"):
            events.append(event)

        obs_events = [e for e in events if e.type == EventType.OBSERVATION]
        assert len(obs_events) == 2

        # 获取每个工具的 duration
        durations = {e.data["tool"]: e.data["duration"] for e in obs_events}

        # fast_tool 应该约 0.05s，slow_tool 应该约 0.15s
        assert durations["fast_tool"] < 0.1, f"fast_tool duration {durations['fast_tool']:.3f}s 应该 < 0.1s"
        assert durations["slow_tool"] > 0.1, f"slow_tool duration {durations['slow_tool']:.3f}s 应该 > 0.1s"
        # 两者差距应该明显
        assert durations["slow_tool"] > durations["fast_tool"] * 2, \
            "slow_tool 耗时应该明显大于 fast_tool"
