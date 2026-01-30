"""SkillRegistry 测试"""

from pathlib import Path

import pytest

from pure_agent_loop.skill.registry import SkillRegistry


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillRegistry:
    """SkillRegistry 测试类"""

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        """创建 Registry 实例"""
        skill_dir = FIXTURES_DIR / "skill-dir"
        return SkillRegistry([skill_dir])

    @pytest.mark.asyncio
    async def test_initialize(self, registry: SkillRegistry):
        """测试初始化加载索引"""
        await registry.initialize()

        assert registry.initialized
        assert registry.size == 2

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, registry: SkillRegistry):
        """测试多次初始化是幂等的"""
        await registry.initialize()
        await registry.initialize()

        assert registry.size == 2

    @pytest.mark.asyncio
    async def test_get_existing_skill(self, registry: SkillRegistry):
        """测试获取已存在的 Skill"""
        await registry.initialize()

        info = registry.get("skill-a")
        assert info is not None
        assert info.name == "skill-a"

    @pytest.mark.asyncio
    async def test_get_nonexistent_skill(self, registry: SkillRegistry):
        """测试获取不存在的 Skill 返回 None"""
        await registry.initialize()

        info = registry.get("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_all(self, registry: SkillRegistry):
        """测试获取所有 Skill"""
        await registry.initialize()

        skills = registry.get_all()
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names

    @pytest.mark.asyncio
    async def test_load_skill_content(self, registry: SkillRegistry):
        """测试加载 Skill 完整内容"""
        await registry.initialize()

        content = await registry.load("skill-a")
        assert content is not None
        assert content.name == "skill-a"
        assert "技能 A" in content.content

    @pytest.mark.asyncio
    async def test_load_nonexistent_skill(self, registry: SkillRegistry):
        """测试加载不存在的 Skill 返回 None"""
        await registry.initialize()

        content = await registry.load("nonexistent")
        assert content is None

