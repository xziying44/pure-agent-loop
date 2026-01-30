"""skill 工具创建"""

from __future__ import annotations

from typing import Any

from ..tool import Tool
from .registry import SkillRegistry


def create_skill_tool(registry: SkillRegistry) -> Tool:
    """创建 skill 工具

    Args:
        registry: SkillRegistry 实例

    Returns:
        配置好的 Tool 对象
    """
    skills = registry.get_all()

    description_lines = [
        "加载技能（Skill）以获取特定任务的详细指导，或执行 Skill 附带的脚本/读取资源文件。",
        "",
        "action 参数说明：",
        "- load: 加载 Skill 知识文档，获取详细指导",
        "- run: 执行 Skill 目录下已声明的脚本",
        "- read_file: 读取 Skill 目录下的资源文件（模板、样式等）",
        "",
        "<available_skills>",
    ]

    for skill in skills:
        description_lines.append("  <skill>")
        description_lines.append(f"    <name>{skill.name}</name>")
        description_lines.append(f"    <description>{skill.description}</description>")
        if skill.scripts:
            description_lines.append("    <scripts>")
            for script in skill.scripts:
                description_lines.append(
                    f"      <script name=\"{script.name}\">{script.description}</script>"
                )
            description_lines.append("    </scripts>")
        description_lines.append("  </skill>")

    description_lines.append("</available_skills>")
    description = "\n".join(description_lines)

    async def execute_skill_action(
        action: str,
        name: str,
        script: str | None = None,
        args: dict[str, Any] | None = None,
        file_path: str | None = None,
    ) -> str:
        """执行 skill 工具操作"""
        if action == "load":
            skill_content = await registry.load(name)
            if not skill_content:
                available = [s.name for s in registry.get_all()]
                return f"错误：Skill '{name}' 不存在。可用 Skill：{available}"

            lines = [
                f"## Skill: {skill_content.name}",
                "",
                f"**基础目录**: {skill_content.base_dir}",
                "",
                "---",
                "",
                skill_content.content.strip(),
            ]
            return "\n".join(lines)

        if action == "run":
            if not script:
                return "错误：action='run' 时必须指定 script 参数"
            return await registry.execute_script(name, script, args)

        if action == "read_file":
            if not file_path:
                return "错误：action='read_file' 时必须指定 file_path 参数"
            return await registry.read_file(name, file_path)

        return f"错误：未知的 action '{action}'，支持的值：load, run, read_file"

    return Tool(
        name="skill",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["load", "run", "read_file"],
                    "description": "操作类型：load（加载知识）、run（执行脚本）、read_file（读取资源）",
                },
                "name": {
                    "type": "string",
                    "description": "Skill 名称（从 available_skills 中选择）",
                },
                "script": {
                    "type": "string",
                    "description": "脚本文件名（action=run 时必需）",
                },
                "args": {
                    "type": "object",
                    "description": "脚本参数（action=run 时可选）",
                },
                "file_path": {
                    "type": "string",
                    "description": "相对于 Skill 目录的文件路径（action=read_file 时必需）",
                },
            },
            "required": ["action", "name"],
        },
        function=execute_skill_action,
        is_async=True,
    )

