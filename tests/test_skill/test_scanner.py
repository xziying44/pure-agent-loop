"""目录扫描器测试"""

from pathlib import Path

import pytest

from pure_agent_loop.skill.scanner import SkillScanner


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillScanner:
    """SkillScanner 测试类"""

    def test_scan_finds_skills(self):
        """测试扫描目录找到所有 SKILL.md"""
        skill_dir = FIXTURES_DIR / "skill-dir"
        scanner = SkillScanner([skill_dir])
        paths = scanner.scan()

        assert len(paths) == 2
        names = [p.parent.name for p in paths]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_scan_empty_directory(self, tmp_path: Path):
        """测试扫描空目录返回空列表"""
        scanner = SkillScanner([tmp_path])
        paths = scanner.scan()
        assert paths == []

    def test_scan_nonexistent_directory(self):
        """测试扫描不存在的目录返回空列表"""
        scanner = SkillScanner([Path("/nonexistent/path")])
        paths = scanner.scan()
        assert paths == []

    def test_scan_multiple_directories(self):
        """测试扫描多个目录"""
        skill_dir = FIXTURES_DIR / "skill-dir"
        valid_skill_dir = FIXTURES_DIR  # 包含 valid-skill
        scanner = SkillScanner([skill_dir, valid_skill_dir])
        paths = scanner.scan()

        # skill-dir 下有 2 个，fixtures 下有 valid-skill 和 invalid-skill
        assert len(paths) >= 2

    def test_scan_deduplicates(self):
        """测试扫描结果去重"""
        skill_dir = FIXTURES_DIR / "skill-dir"
        scanner = SkillScanner([skill_dir, skill_dir])  # 重复添加
        paths = scanner.scan()

        assert len(paths) == 2  # 应该去重

