"""内置系统提示词模板测试"""

import pytest
from pure_agent_loop.prompts import build_system_prompt


class TestBuildSystemPrompt:
    """build_system_prompt 测试"""

    def test_default_name(self):
        """默认名称应为 '智能助理'"""
        prompt = build_system_prompt()
        assert "你是智能助理" in prompt

    def test_custom_name(self):
        """自定义名称应注入到角色描述"""
        prompt = build_system_prompt(name="研究助手")
        assert "你是研究助手" in prompt

    def test_user_prompt_injected(self):
        """用户自定义提示词应被注入"""
        prompt = build_system_prompt(user_prompt="你擅长数学计算。")
        assert "你擅长数学计算。" in prompt

    def test_empty_user_prompt(self):
        """空用户提示词不应导致异常"""
        prompt = build_system_prompt(user_prompt="")
        assert "你是智能助理" in prompt

    def test_contains_todo_requirement(self):
        """提示词应包含 TodoWrite 使用要求"""
        prompt = build_system_prompt()
        assert "todo_write" in prompt

    def test_contains_react_guidance(self):
        """提示词应包含思考-行动指导"""
        prompt = build_system_prompt()
        assert "思考" in prompt
        assert "行动" in prompt

    def test_contains_role_section(self):
        """提示词应包含角色描述段"""
        prompt = build_system_prompt()
        assert "# Role" in prompt or "# 角色" in prompt
