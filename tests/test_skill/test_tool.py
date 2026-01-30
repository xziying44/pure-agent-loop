"""skill 工具测试"""

from pathlib import Path

import pytest
import pytest_asyncio

from pure_agent_loop.skill.registry import SkillRegistry
from pure_agent_loop.skill.tool import create_skill_tool


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillTool:
    """skill 工具测试类"""

    @pytest_asyncio.fixture
    async def registry(self) -> SkillRegistry:
        """创建 Registry"""
        reg = SkillRegistry([FIXTURES_DIR])
        await reg.initialize()
        return reg

    @pytest.fixture
    def skill_tool(self, registry: SkillRegistry):
        """创建 skill 工具"""
        return create_skill_tool(registry)

    def test_tool_name(self, skill_tool):
        """测试工具名称"""
        assert skill_tool.name == "skill"

    def test_tool_description_contains_skills(self, skill_tool):
        """测试工具描述包含可用 Skill 列表"""
        desc = skill_tool.description
        assert "skill-a" in desc
        assert "skill-b" in desc
        assert "script-skill" in desc

    def test_tool_parameters(self, skill_tool):
        """测试工具参数定义"""
        params = skill_tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "name" in params["properties"]
        assert params["properties"]["action"]["enum"] == ["load", "run", "read_file"]

    @pytest.mark.asyncio
    async def test_execute_load_action(self, skill_tool):
        """测试 load 操作"""
        result = await skill_tool.execute({"action": "load", "name": "skill-a"})

        assert "skill-a" in result
        assert "技能 A" in result

    @pytest.mark.asyncio
    async def test_execute_load_nonexistent(self, skill_tool):
        """测试 load 不存在的 Skill"""
        result = await skill_tool.execute({"action": "load", "name": "nonexistent"})
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_execute_read_file_action(self, skill_tool):
        """测试 read_file 操作"""
        result = await skill_tool.execute(
            {
                "action": "read_file",
                "name": "script-skill",
                "file_path": "templates/sample.txt",
            }
        )

        assert "示例模板文件" in result

    @pytest.mark.asyncio
    async def test_execute_run_action(self, skill_tool):
        """测试 run 操作"""
        result = await skill_tool.execute(
            {
                "action": "run",
                "name": "script-skill",
                "script": "hello.py",
                "args": {"name": "Claude"},
            }
        )

        assert "Hello, Claude!" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, skill_tool):
        """测试未知操作"""
        result = await skill_tool.execute({"action": "unknown", "name": "skill-a"})

        assert "错误" in result
        assert "未知" in result

