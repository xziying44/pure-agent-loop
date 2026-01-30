"""Frontmatter 解析器测试"""

from pathlib import Path

import pytest

from pure_agent_loop.skill.parser import SkillParser


# 获取 fixtures 目录路径
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillParser:
    """SkillParser 测试类"""

    def test_parse_frontmatter_valid(self):
        """测试解析有效的 SKILL.md"""
        skill_file = FIXTURES_DIR / "valid-skill" / "SKILL.md"
        info = SkillParser.parse_frontmatter(skill_file)

        assert info is not None
        assert info.name == "test-skill"
        assert info.description == "这是一个用于测试的技能"
        assert len(info.scripts) == 1
        assert info.scripts[0].name == "hello.py"
        assert info.scripts[0].description == "打印问候语"
        assert info.base_dir == skill_file.parent.resolve()

    def test_parse_frontmatter_missing_description(self):
        """测试缺少必需字段时返回 None"""
        skill_file = FIXTURES_DIR / "invalid-skill" / "SKILL.md"
        info = SkillParser.parse_frontmatter(skill_file)

        assert info is None

    def test_parse_frontmatter_file_not_found(self):
        """测试文件不存在时返回 None"""
        skill_file = FIXTURES_DIR / "nonexistent" / "SKILL.md"
        info = SkillParser.parse_frontmatter(skill_file)

        assert info is None

    def test_parse_full_valid(self):
        """测试解析完整内容"""
        skill_file = FIXTURES_DIR / "valid-skill" / "SKILL.md"
        info, content = SkillParser.parse_full(skill_file)

        assert info is not None
        assert info.name == "test-skill"
        assert "测试技能" in content
        assert "使用说明" in content

    def test_parse_frontmatter_no_scripts(self, tmp_path: Path):
        """测试没有脚本声明的 SKILL.md"""
        md = tmp_path / "SKILL.md"
        md.write_text(
            """---
name: no-scripts-skill
description: 没有脚本的技能
---

# 内容
""",
            encoding="utf-8",
        )

        info = SkillParser.parse_frontmatter(md)
        assert info is not None
        assert info.name == "no-scripts-skill"
        assert info.scripts == []

