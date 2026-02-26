"""系统提示词测试"""

from pure_agent_loop.prompts import build_system_prompt


class TestBuildSystemPrompt:
    """build_system_prompt 测试"""

    def test_default_name(self):
        """默认名称应为 '智能助理'"""
        prompt = build_system_prompt()
        assert "智能助理" in prompt

    def test_custom_name(self):
        """自定义名称应注入"""
        prompt = build_system_prompt(name="研究助手")
        assert "研究助手" in prompt

    def test_user_prompt_injected(self):
        """用户提示词应注入"""
        prompt = build_system_prompt(user_prompt="你擅长数学。")
        assert "你擅长数学。" in prompt

    def test_empty_user_prompt(self):
        """空用户提示词不应异常"""
        prompt = build_system_prompt(user_prompt="")
        assert "智能助理" in prompt

    def test_contains_objectivity_section(self):
        """应包含专业客观性段落"""
        prompt = build_system_prompt()
        assert "客观" in prompt

    def test_contains_todo_guidance(self):
        """应包含 todo_write 使用引导"""
        prompt = build_system_prompt()
        assert "todo_write" in prompt

    def test_contains_skill_guidance(self):
        """应包含技能系统引导"""
        prompt = build_system_prompt()
        assert "技能" in prompt or "skill" in prompt.lower()

    def test_contains_tool_usage_guidance(self):
        """应包含工具使用策略"""
        prompt = build_system_prompt()
        assert "并行" in prompt

    def test_contains_output_style_guidance(self):
        """应包含输出风格指南"""
        prompt = build_system_prompt()
        assert "简洁" in prompt

    def test_no_mandatory_protocol_language(self):
        """不应包含旧的强制协议用语"""
        prompt = build_system_prompt()
        assert "唯一起手式" not in prompt
        assert "SOP" not in prompt
        assert "严禁" not in prompt
        assert "综合研判" not in prompt

    def test_user_section_at_end(self):
        """用户自定义指令应在提示词末尾"""
        prompt = build_system_prompt(user_prompt="自定义内容")
        # 用户内容应在提示词的后半部分
        idx = prompt.index("自定义内容")
        assert idx > len(prompt) // 2
