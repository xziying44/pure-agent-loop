"""Skill 系统数据类型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillScript:
    """Skill 附带脚本的元数据

    Attributes:
        name: 脚本文件名
        description: 脚本功能描述
        args_schema: 参数 JSON Schema
    """

    name: str
    description: str
    args_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillInfo:
    """Skill 索引信息（轻量，启动时加载）

    Attributes:
        name: 唯一标识符
        description: 供 LLM 判断使用场景的描述
        location: SKILL.md 绝对路径
        base_dir: Skill 所在目录（用于资源定位）
        scripts: 可执行脚本列表
    """

    name: str
    description: str
    location: Path
    base_dir: Path
    scripts: list[SkillScript] = field(default_factory=list)


@dataclass
class SkillContent:
    """Skill 完整内容（按需加载）

    继承 SkillInfo 的所有字段，额外包含 Markdown 正文。

    Attributes:
        name: 唯一标识符
        description: 供 LLM 判断使用场景的描述
        location: SKILL.md 绝对路径
        base_dir: Skill 所在目录
        scripts: 可执行脚本列表
        content: Markdown 正文（知识指导）
    """

    name: str
    description: str
    location: Path
    base_dir: Path
    scripts: list[SkillScript] = field(default_factory=list)
    content: str = ""

