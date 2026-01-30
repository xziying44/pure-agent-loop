"""Agent 入口测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pure_agent_loop.agent import Agent, AgentResult
from pure_agent_loop.tool import tool
from pure_agent_loop.events import Event, EventType
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage
from pure_agent_loop.limits import LoopLimits
from pure_agent_loop.retry import RetryConfig


class MockLLM(BaseLLMClient):
    """测试用 Mock LLM"""

    def __init__(self, responses):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _text_response(content):
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        raw={},
    )


def _tool_call_response(name, args):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=args)],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )


class TestAgent:
    """Agent 入口测试"""

    @pytest.mark.asyncio
    async def test_arun_simple(self):
        """异步执行应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("打个招呼")

        assert isinstance(result, AgentResult)
        assert result.content == "你好"
        assert result.stop_reason == "completed"
        assert result.steps >= 1
        assert len(result.events) > 0
        assert len(result.messages) > 0

    def test_run_simple(self):
        """同步执行应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = agent.run("打个招呼")

        assert isinstance(result, AgentResult)
        assert result.content == "你好"

    @pytest.mark.asyncio
    async def test_arun_with_tools(self):
        """带工具执行应正确处理"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        mock_llm = MockLLM([
            _tool_call_response("add", {"a": 1, "b": 2}),
            _text_response("1 + 2 = 3"),
        ])

        agent = Agent(llm=mock_llm, tools=[add])
        result = await agent.arun("计算 1+2")

        assert result.content == "1 + 2 = 3"
        assert result.stop_reason == "completed"
        # 应该有工具调用事件
        action_events = [e for e in result.events if e.type == EventType.ACTION]
        assert len(action_events) == 1

    @pytest.mark.asyncio
    async def test_arun_stream(self):
        """流式执行应逐步产出事件"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)

        events = []
        async for event in agent.arun_stream("打个招呼"):
            events.append(event)

        assert len(events) > 0
        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

    def test_run_stream(self):
        """同步流式执行应逐步产出事件"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)

        events = list(agent.run_stream("打个招呼"))
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """多轮对话应传递消息历史"""
        mock_llm1 = MockLLM([_text_response("北京今天晴天")])
        agent1 = Agent(llm=mock_llm1)
        r1 = await agent1.arun("北京天气")

        mock_llm2 = MockLLM([_text_response("上海今天多云")])
        agent2 = Agent(llm=mock_llm2)
        r2 = await agent2.arun("那上海呢？", messages=r1.messages)

        assert r2.content == "上海今天多云"
        # 消息历史应该包含前一轮的内容
        assert len(r2.messages) > len(r1.messages)

    def test_constructor_with_model_params(self):
        """应该支持 model 参数构造"""
        agent = Agent(
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        assert agent is not None

    def test_constructor_with_custom_limits(self):
        """应该支持自定义限制"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test",
            limits=LoopLimits(max_steps=5, timeout=60.0),
        )
        assert agent is not None


class TestAgentResult:
    """AgentResult 测试"""

    def test_create_result(self):
        """应该能创建 AgentResult"""
        result = AgentResult(
            content="测试",
            steps=3,
            total_tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            events=[],
            stop_reason="completed",
            messages=[],
        )
        assert result.content == "测试"
        assert result.steps == 3


class TestAgentName:
    """Agent name 参数测试"""

    @pytest.mark.asyncio
    async def test_default_name(self):
        """默认名称应为 '智能助理'"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        # Agent 内部应使用默认名称构建提示词
        assert agent._name == "智能助理"

    @pytest.mark.asyncio
    async def test_custom_name(self):
        """自定义名称应被保存"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm, name="研究助手")
        assert agent._name == "研究助手"


class TestAgentTodoIntegration:
    """Agent TodoWrite 集成测试"""

    @pytest.mark.asyncio
    async def test_todo_write_auto_registered(self):
        """todo_write 工具应被自动注册"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        # 工具注册表应包含 todo_write
        assert agent._tool_registry.get("todo_write") is not None

    @pytest.mark.asyncio
    async def test_agent_result_has_todos(self):
        """AgentResult 应包含 todos 属性"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("打个招呼")
        assert hasattr(result, "todos")
        assert isinstance(result.todos, list)

    @pytest.mark.asyncio
    async def test_todo_write_updates_result(self):
        """通过 todo_write 工具更新的任务应出现在 AgentResult.todos"""
        mock_llm = MockLLM([
            _tool_call_response("todo_write", {
                "todos": [
                    {"content": "步骤1", "status": "completed"},
                    {"content": "步骤2", "status": "pending"},
                ]
            }),
            _text_response("任务已规划"),
        ])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("规划任务")
        assert len(result.todos) == 2
        assert result.todos[0]["content"] == "步骤1"
        assert result.todos[1]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_user_tools_preserved(self):
        """用户注册的工具不应被内置工具覆盖"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return "结果"

        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm, tools=[search])
        assert agent._tool_registry.get("search") is not None
        assert agent._tool_registry.get("todo_write") is not None


class TestAgentThinkingMode:
    """Agent 思考模式测试"""

    def test_default_thinking_level_is_off(self):
        """默认 thinking_level 应该是 off"""
        agent = Agent(model="gpt-4o-mini", api_key="test-key")
        assert agent._thinking_level == "off"

    def test_thinking_level_can_be_set(self):
        """应该能设置 thinking_level"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            thinking_level="high",
        )
        assert agent._thinking_level == "high"

    def test_default_emit_reasoning_events_is_false(self):
        """默认 emit_reasoning_events 应该是 False"""
        agent = Agent(model="gpt-4o-mini", api_key="test-key")
        assert agent._emit_reasoning_events is False

    def test_emit_reasoning_events_can_be_enabled(self):
        """应该能启用 emit_reasoning_events"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            emit_reasoning_events=True,
        )
        assert agent._emit_reasoning_events is True

    def test_thinking_level_passed_to_openai_client(self):
        """thinking_level 应该传递给 OpenAIClient"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            thinking_level="medium",
        )
        # 验证内部 LLM 客户端的 thinking_level
        assert agent._llm.thinking_level == "medium"
