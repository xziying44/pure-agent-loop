"""SKILL.md 文件解析器"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .types import SkillInfo, SkillScript

logger = logging.getLogger(__name__)


class SkillParser:
    """SKILL.md 文件解析器

    支持解析 YAML frontmatter 和 Markdown 正文。
    """

    DELIMITER = "---"

    @classmethod
    def _split_frontmatter(cls, raw: str) -> tuple[dict[str, Any] | None, str | None]:
        """拆分 frontmatter 与正文

        Args:
            raw: SKILL.md 全文

        Returns:
            (metadata, body)；若不符合 frontmatter 格式则返回 (None, None)
        """
        lines = raw.splitlines()
        if not lines:
            return None, None

        if lines[0].strip() != cls.DELIMITER:
            return None, None

        end_idx: int | None = None
        for i in range(1, len(lines)):
            if lines[i].strip() == cls.DELIMITER:
                end_idx = i
                break

        if end_idx is None:
            return None, None

        yaml_str = "\n".join(lines[1:end_idx]).strip()
        try:
            metadata = yaml.safe_load(yaml_str) if yaml_str else {}
        except yaml.YAMLError:
            return None, None

        if not isinstance(metadata, dict):
            return None, None

        body = "\n".join(lines[end_idx + 1 :]).strip()
        return metadata, body

    @classmethod
    def parse_frontmatter(cls, file_path: Path) -> SkillInfo | None:
        """解析 SKILL.md 文件的 frontmatter

        仅读取元数据，不读取正文内容（懒加载）。

        Args:
            file_path: SKILL.md 文件路径

        Returns:
            SkillInfo 或 None（解析失败时）
        """
        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception:
            logger.debug("读取 SKILL.md 失败: %s", file_path, exc_info=True)
            return None

        metadata, _ = cls._split_frontmatter(raw)
        if not metadata:
            return None

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(description, str) or not description.strip():
            return None

        scripts: list[SkillScript] = []
        scripts_data = metadata.get("scripts") or []
        if isinstance(scripts_data, list):
            for item in scripts_data:
                if not isinstance(item, dict):
                    continue
                script_name = item.get("name") or ""
                script_desc = item.get("description") or ""
                args_schema = item.get("args_schema") or {}
                if not isinstance(args_schema, dict):
                    args_schema = {}

                scripts.append(
                    SkillScript(
                        name=str(script_name),
                        description=str(script_desc),
                        args_schema=args_schema,
                    )
                )

        return SkillInfo(
            name=name.strip(),
            description=description.strip(),
            location=file_path.resolve(),
            base_dir=file_path.parent.resolve(),
            scripts=scripts,
        )

    @classmethod
    def parse_full(cls, file_path: Path) -> tuple[SkillInfo | None, str]:
        """解析 SKILL.md 完整内容（frontmatter + 正文）

        Args:
            file_path: SKILL.md 文件路径

        Returns:
            (SkillInfo, content) 或 (None, "")
        """
        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception:
            logger.debug("读取 SKILL.md 失败: %s", file_path, exc_info=True)
            return None, ""

        info = cls.parse_frontmatter(file_path)
        if not info:
            return None, ""

        _, body = cls._split_frontmatter(raw)
        return info, body or ""

