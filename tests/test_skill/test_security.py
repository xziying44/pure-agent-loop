"""Skill 安全机制测试"""

from pathlib import Path

import pytest
import pytest_asyncio

from pure_agent_loop.skill.registry import SkillRegistry


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillSecurity:
    """安全机制测试类"""

    @pytest_asyncio.fixture
    async def registry(self) -> SkillRegistry:
        """创建包含脚本 Skill 的 Registry"""
        reg = SkillRegistry([FIXTURES_DIR])
        await reg.initialize()
        return reg

    # ===== read_file 测试 =====

    @pytest.mark.asyncio
    async def test_read_file_success(self, registry: SkillRegistry):
        """测试成功读取资源文件"""
        result = await registry.read_file("script-skill", "templates/sample.txt")

        assert "示例模板文件" in result
        assert "错误" not in result

    @pytest.mark.asyncio
    async def test_read_file_nonexistent_skill(self, registry: SkillRegistry):
        """测试读取不存在 Skill 的文件"""
        result = await registry.read_file("nonexistent", "file.txt")

        assert "错误" in result
        assert "不存在" in result

    @pytest.mark.asyncio
    async def test_read_file_nonexistent_file(self, registry: SkillRegistry):
        """测试读取不存在的文件"""
        result = await registry.read_file("script-skill", "nonexistent.txt")

        assert "错误" in result
        assert "不存在" in result

    @pytest.mark.asyncio
    async def test_read_file_path_traversal_blocked(self, registry: SkillRegistry):
        """测试路径遍历攻击被阻止"""
        result = await registry.read_file("script-skill", "../valid-skill/SKILL.md")

        assert "错误" in result
        assert "越界" in result

    @pytest.mark.asyncio
    async def test_read_file_absolute_path_blocked(self, registry: SkillRegistry):
        """测试绝对路径被正确处理"""
        result = await registry.read_file("script-skill", "/etc/passwd")
        assert "错误" in result

    # ===== execute_script 测试 =====

    @pytest.mark.asyncio
    async def test_execute_script_success(self, registry: SkillRegistry):
        """测试成功执行脚本"""
        result = await registry.execute_script(
            "script-skill", "hello.py", {"name": "Test"}
        )

        assert "Hello, Test!" in result

    @pytest.mark.asyncio
    async def test_execute_script_default_args(self, registry: SkillRegistry):
        """测试使用默认参数执行脚本"""
        result = await registry.execute_script("script-skill", "hello.py", None)

        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_execute_script_nonexistent_skill(self, registry: SkillRegistry):
        """测试执行不存在 Skill 的脚本"""
        result = await registry.execute_script("nonexistent", "hello.py", None)

        assert "错误" in result
        assert "不存在" in result

    @pytest.mark.asyncio
    async def test_execute_script_undeclared_script(self, registry: SkillRegistry):
        """测试执行未声明的脚本被拒绝"""
        result = await registry.execute_script("script-skill", "undeclared.py", None)

        assert "错误" in result
        assert "未在" in result or "未声明" in result

    @pytest.mark.asyncio
    async def test_execute_script_path_traversal_blocked(self, registry: SkillRegistry):
        """测试脚本路径遍历被阻止"""
        result = await registry.execute_script("script-skill", "../other/script.py", None)
        assert "错误" in result
