"""Agent Skill 集成测试"""

from pathlib import Path

import pytest

from pure_agent_loop import Agent
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class MockLLM(BaseLLMClient):
    """Mock LLM 客户端"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


class TestAgentSkillIntegration:
    """Agent Skill 集成测试"""

    @pytest.mark.asyncio
    async def test_agent_with_skills_dir(self):
        """测试 Agent 使用 skills_dir 参数"""
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="skill",
                        arguments={"action": "load", "name": "skill-a"},
                    )
                ],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                raw={},
            ),
            LLMResponse(
                content="根据技能 A 的指导，任务已完成。",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                raw={},
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(
            llm=mock_llm,
            skills_dir=str(FIXTURES_DIR / "skill-dir"),
        )

        result = await agent.arun("请使用技能 A 完成任务")

        assert result.stop_reason == "completed"
        assert "任务已完成" in result.content

    @pytest.mark.asyncio
    async def test_agent_skill_tool_registered(self):
        """测试 skill 工具被正确注册"""
        responses = [
            LLMResponse(
                content="完成",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                raw={},
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(
            llm=mock_llm,
            skills_dir=str(FIXTURES_DIR / "skill-dir"),
        )

        await agent.arun("test")

        skill_tool = agent._tool_registry.get("skill")
        assert skill_tool is not None
        assert "skill-a" in skill_tool.description
        assert "skill-b" in skill_tool.description

    @pytest.mark.asyncio
    async def test_agent_without_skills_dir(self):
        """测试不使用 skills_dir 时正常工作"""
        responses = [
            LLMResponse(
                content="完成",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                raw={},
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(llm=mock_llm)

        result = await agent.arun("test")

        assert result.stop_reason == "completed"
        skill_tool = agent._tool_registry.get("skill")
        assert skill_tool is None

