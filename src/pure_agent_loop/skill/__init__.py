"""Skill 动态知识注入系统

支持按需加载专业领域知识、执行脚本和读取资源文件。
"""

from .types import SkillContent, SkillInfo, SkillScript
from .parser import SkillParser
from .scanner import SkillScanner
from .registry import SkillRegistry
from .tool import create_skill_tool

__all__ = [
    "SkillContent",
    "SkillInfo",
    "SkillScript",
    "SkillParser",
    "SkillScanner",
    "SkillRegistry",
    "create_skill_tool",
]
