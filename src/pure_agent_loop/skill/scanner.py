"""Skill 目录扫描器"""

from __future__ import annotations

from pathlib import Path


class SkillScanner:
    """Skill 目录扫描器

    在指定目录中发现所有 SKILL.md 文件。
    支持的目录结构：skills/xxx/SKILL.md
    """

    SKILL_FILE = "SKILL.md"

    def __init__(self, directories: list[Path]):
        """
        Args:
            directories: 要扫描的目录列表
        """
        self.directories = directories

    def scan(self) -> list[Path]:
        """扫描所有目录，返回发现的 SKILL.md 路径列表

        Returns:
            SKILL.md 文件的绝对路径列表（已去重，顺序稳定）
        """
        results: set[Path] = set()

        for base_dir in self.directories:
            if not base_dir.exists():
                continue

            # 递归扫描，允许用户按目录层级组织技能（例如 skills/web/search/SKILL.md）
            for skill_file in base_dir.rglob(self.SKILL_FILE):
                if skill_file.is_file():
                    results.add(skill_file.resolve())

        return sorted(results, key=lambda p: str(p))
