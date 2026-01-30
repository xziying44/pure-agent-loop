# Skill 动态知识注入系统 - 实施设计文档

> **项目**: pure-agent-loop
> **日期**: 2026-01-30
> **状态**: 待实施

---

## 1. 概述

### 1.1 背景

当前项目中，工具使用策略和领域知识需要硬编码在 `system_prompt` 中（如 `examples/exa_search.py`）。这导致：
- 系统提示词臃肿，难以维护
- 知识无法按需加载，浪费上下文窗口
- 相同知识在不同 Agent 间难以复用

### 1.2 目标

实现 Skill 系统，允许为智能体提供**按需加载的专业领域知识**，同时支持执行 Skill 附带的脚本和读取资源文件。

### 1.3 核心特性

| 特性 | 说明 |
|-----|------|
| 声明式配置 | 使用 Markdown + YAML frontmatter 定义 Skill |
| 自动发现 | 多目录扫描，自动索引所有可用 Skill |
| 懒加载 | 启动时只加载 frontmatter 索引，完整内容按需读取 |
| LLM 驱动选择 | 利用大模型语义理解能力自主选择合适的 Skill |
| 白名单安全 | 脚本执行和文件读取严格限制在 Skill 目录内 |

---

## 2. 系统架构

### 2.1 模块关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent                                    │
│  新增参数: skills_dir: str | list[str]                          │
│  初始化时自动扫描目录，构建 SkillRegistry                        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SkillRegistry（新增）                         │
│  • skills: dict[name, SkillInfo]  — 元数据索引                  │
│  • base_dirs: list[Path]          — 合法路径白名单              │
│  • get_all() -> list[SkillInfo]   — 获取所有摘要                │
│  • load(name) -> SkillContent     — 按需加载完整内容            │
│  • execute_script(name, script, args) — 安全执行脚本            │
│  • read_file(name, path) -> str   — 安全读取资源文件            │
└─────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    skill 工具（新增）                            │
│  参数:                                                           │
│    action: "load" | "run" | "read_file"                         │
│    name: skill 名称                                             │
│    script: 脚本文件名（仅 run 需要）                             │
│    args: 脚本参数（可选）                                        │
│    file_path: 相对文件路径（仅 read_file 需要）                  │
│                                                                  │
│  tool description 动态包含所有可用 skill 列表                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户输入: "帮我搜索关于 AI 的最新资讯"
                    │
                    ▼
┌─────────────────────────────────────────┐
│            LLM 推理引擎                  │
│  收到可用工具列表，包括:                 │
│  - skill tool                           │
│    - available: web-search              │
│    - available: code-review             │
│  - exa_search                           │
│  - fetch_webpage                        │
└────────────────┬────────────────────────┘
                 │ 决策：先调用 skill tool 加载知识
                 │ 参数：{action: "load", name: "web-search"}
                 ▼
┌─────────────────────────────────────────┐
│           skill tool.execute()          │
│  1. 查询 SkillRegistry                  │
│  2. 获取 SKILL.md 路径                  │
│  3. 读取完整文件内容                    │
│  4. 返回格式化结果                      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            LLM 继续推理                  │
│  基于 Skill 内容中的专业指导            │
│  调用 exa_search → fetch_webpage        │
│  完成用户的搜索任务                     │
└─────────────────────────────────────────┘
```

---

## 3. 数据结构定义

### 3.1 SKILL.md 文件格式

```markdown
---
name: skill-identifier
description: 一句话描述这个 Skill 的用途和适用场景（供 LLM 判断何时使用）
scripts:
  - name: script-name.py
    description: 脚本功能描述
    args_schema:
      param1: { type: string, required: true, description: "参数描述" }
      param2: { type: integer, default: 10 }
---

# Skill 标题

## 适用场景
- 场景 1
- 场景 2

## 核心知识 / 工具使用策略
[详细内容...]

## 资源文件
- 模板文件：`templates/xxx.html`
- 参考代码：`examples/xxx.py`

## 注意事项
- 注意点 1
- 注意点 2
```

**规范**：
- 文件名必须是 `SKILL.md`（大写）
- frontmatter 中 `name` 和 `description` 是必需字段
- `scripts` 字段可选，用于声明可执行脚本的白名单
- `description` 应清晰描述**何时使用**此 Skill，以 "当...时使用" 格式描述

### 3.2 Python 数据类

```python
# src/pure_agent_loop/skill/types.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillScript:
    """Skill 附带脚本的元数据"""
    name: str                     # 脚本文件名
    description: str              # 脚本功能描述
    args_schema: dict[str, Any]   # 参数 JSON Schema

    @property
    def path(self) -> Path | None:
        """脚本路径（由 Registry 在运行时设置）"""
        return getattr(self, '_path', None)

    @path.setter
    def path(self, value: Path) -> None:
        self._path = value


@dataclass
class SkillInfo:
    """Skill 索引信息（轻量，启动时加载）"""
    name: str                     # 唯一标识符
    description: str              # 供 LLM 判断使用场景的描述
    location: Path                # SKILL.md 绝对路径
    base_dir: Path                # Skill 所在目录（用于资源定位）
    scripts: list[SkillScript] = field(default_factory=list)


@dataclass
class SkillContent(SkillInfo):
    """Skill 完整内容（按需加载）"""
    content: str = ""             # Markdown 正文（知识指导）
```

---

## 4. 核心模块实现

### 4.1 Frontmatter 解析器

```python
# src/pure_agent_loop/skill/parser.py

import yaml
from pathlib import Path
from .types import SkillInfo, SkillScript


class SkillParser:
    """SKILL.md 文件解析器"""

    DELIMITER = "---"

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
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[SkillParser] 读取文件失败: {file_path} - {e}")
            return None

        # 检查是否以 --- 开头
        if not content.startswith(cls.DELIMITER):
            print(f"[SkillParser] 缺少 frontmatter: {file_path}")
            return None

        # 查找结束的 ---
        end_index = content.find(cls.DELIMITER, 3)
        if end_index == -1:
            print(f"[SkillParser] frontmatter 未闭合: {file_path}")
            return None

        # 提取 YAML 部分
        yaml_str = content[3:end_index].strip()

        try:
            metadata = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            print(f"[SkillParser] YAML 解析失败: {file_path} - {e}")
            return None

        # 验证必需字段
        if not metadata or not metadata.get("name") or not metadata.get("description"):
            print(f"[SkillParser] 缺少必需字段 (name/description): {file_path}")
            return None

        # 解析脚本列表
        scripts = []
        for script_data in metadata.get("scripts", []):
            scripts.append(SkillScript(
                name=script_data.get("name", ""),
                description=script_data.get("description", ""),
                args_schema=script_data.get("args_schema", {}),
            ))

        base_dir = file_path.parent

        return SkillInfo(
            name=metadata["name"],
            description=metadata["description"],
            location=file_path,
            base_dir=base_dir,
            scripts=scripts,
        )

    @classmethod
    def parse_full(cls, file_path: Path) -> tuple[SkillInfo | None, str]:
        """解析 SKILL.md 完整内容（frontmatter + 正文）

        Returns:
            (SkillInfo, content) 或 (None, "")
        """
        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[SkillParser] 读取文件失败: {file_path} - {e}")
            return None, ""

        # 解析 frontmatter
        info = cls.parse_frontmatter(file_path)
        if not info:
            return None, ""

        # 提取正文
        end_index = raw.find(cls.DELIMITER, 3)
        content = raw[end_index + 3:].strip()

        return info, content
```

### 4.2 目录扫描器

```python
# src/pure_agent_loop/skill/scanner.py

import glob
from pathlib import Path


class SkillScanner:
    """Skill 目录扫描器"""

    SKILL_FILE = "SKILL.md"

    def __init__(self, directories: list[Path]):
        """
        Args:
            directories: 要扫描的目录列表
        """
        self.directories = directories

    def scan(self) -> list[Path]:
        """扫描所有目录，返回发现的 SKILL.md 路径列表

        支持的目录结构：
        - skills/xxx/SKILL.md
        - skills/SKILL.md
        """
        results: list[Path] = []

        for base_dir in self.directories:
            if not base_dir.exists():
                continue

            # 方式 1：直接子目录下的 SKILL.md
            for skill_file in base_dir.glob(f"*/{self.SKILL_FILE}"):
                results.append(skill_file.resolve())

            # 方式 2：深层嵌套（可选，按需启用）
            # for skill_file in base_dir.rglob(self.SKILL_FILE):
            #     results.append(skill_file.resolve())

        # 去重
        return list(set(results))
```

### 4.3 SkillRegistry 核心

```python
# src/pure_agent_loop/skill/registry.py

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from .types import SkillInfo, SkillContent, SkillScript
from .parser import SkillParser
from .scanner import SkillScanner


class SkillRegistry:
    """Skill 注册表

    负责扫描、索引、加载和执行 Skill。
    使用白名单路径策略确保安全。
    """

    def __init__(self, directories: list[Path]):
        """
        Args:
            directories: 允许的 Skill 目录列表（白名单基准）
        """
        self._base_dirs = [d.resolve() for d in directories]
        self._skills: dict[str, SkillInfo] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def size(self) -> int:
        return len(self._skills)

    async def initialize(self) -> None:
        """扫描目录，构建索引（仅加载 frontmatter）"""
        async with self._init_lock:
            if self._initialized:
                return

            scanner = SkillScanner(self._base_dirs)
            paths = scanner.scan()

            print(f"[SkillRegistry] 发现 {len(paths)} 个 SKILL.md 文件")

            for file_path in paths:
                info = SkillParser.parse_frontmatter(file_path)
                if not info:
                    continue

                # 检测重复
                if info.name in self._skills:
                    existing = self._skills[info.name]
                    print(
                        f"[SkillRegistry] 重复的 Skill 名称: {info.name}\n"
                        f"  已存在: {existing.location}\n"
                        f"  被忽略: {file_path}"
                    )
                    continue

                # 设置脚本路径
                for script in info.scripts:
                    script.path = info.base_dir / "scripts" / script.name

                self._skills[info.name] = info
                print(f"[SkillRegistry] 已注册: {info.name}")

            self._initialized = True

    def get(self, name: str) -> SkillInfo | None:
        """获取单个 Skill 信息"""
        return self._skills.get(name)

    def get_all(self) -> list[SkillInfo]:
        """获取所有 Skill 信息"""
        return list(self._skills.values())

    async def load(self, name: str) -> SkillContent | None:
        """按需加载 Skill 完整内容

        Args:
            name: Skill 名称

        Returns:
            SkillContent 或 None（不存在时）
        """
        info = self._skills.get(name)
        if not info:
            return None

        _, content = SkillParser.parse_full(info.location)

        return SkillContent(
            name=info.name,
            description=info.description,
            location=info.location,
            base_dir=info.base_dir,
            scripts=info.scripts,
            content=content,
        )

    async def read_file(self, skill_name: str, file_path: str) -> str:
        """安全读取 Skill 目录下的资源文件

        安全检查：
        1. skill_name 必须已注册
        2. 最终路径必须位于 skill 目录内（防止 ../ 攻击）

        Args:
            skill_name: Skill 名称
            file_path: 相对于 skill 目录的文件路径

        Returns:
            文件内容或错误信息
        """
        info = self._skills.get(skill_name)
        if not info:
            return f"错误：Skill '{skill_name}' 不存在"

        # 构建绝对路径并规范化
        target_path = (info.base_dir / file_path).resolve()

        # 安全检查：确保路径在 skill 目录内
        if not self._is_path_safe(target_path, info.base_dir):
            return f"错误：路径 '{file_path}' 越界，禁止访问 Skill 目录外的文件"

        # 检查文件是否存在
        if not target_path.exists():
            return f"错误：文件 '{file_path}' 不存在"

        if not target_path.is_file():
            return f"错误：'{file_path}' 不是文件"

        try:
            return target_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"错误：读取文件失败 - {e}"

    async def execute_script(
        self,
        skill_name: str,
        script_name: str,
        args: dict[str, Any] | None = None,
    ) -> str:
        """安全执行 Skill 脚本

        安全检查：
        1. skill_name 必须已注册
        2. script_name 必须在该 skill 的 scripts 列表中声明
        3. 脚本路径必须位于 skill 目录内

        Args:
            skill_name: Skill 名称
            script_name: 脚本文件名
            args: 脚本参数

        Returns:
            执行结果或错误信息
        """
        info = self._skills.get(skill_name)
        if not info:
            return f"错误：Skill '{skill_name}' 不存在"

        # 检查脚本是否在白名单中声明
        script = next((s for s in info.scripts if s.name == script_name), None)
        if not script:
            declared = [s.name for s in info.scripts]
            return (
                f"错误：脚本 '{script_name}' 未在 Skill '{skill_name}' 中声明。\n"
                f"已声明的脚本：{declared}"
            )

        # 构建脚本路径
        script_path = (info.base_dir / "scripts" / script_name).resolve()

        # 安全检查：确保路径在 skill 目录内
        if not self._is_path_safe(script_path, info.base_dir):
            return f"错误：脚本路径越界，禁止执行"

        # 检查脚本是否存在
        if not script_path.exists():
            return f"错误：脚本文件 '{script_path}' 不存在"

        # 执行脚本
        try:
            cmd = ["python", str(script_path)]

            # 将参数作为命令行参数传递
            if args:
                for key, value in args.items():
                    cmd.extend([f"--{key}", str(value)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(info.base_dir),  # 在 skill 目录下执行
            )

            output = result.stdout
            if result.returncode != 0:
                output += f"\n[stderr]: {result.stderr}"

            return output or "(脚本执行完成，无输出)"

        except subprocess.TimeoutExpired:
            return "错误：脚本执行超时（60秒）"
        except Exception as e:
            return f"错误：脚本执行失败 - {e}"

    def _is_path_safe(self, target: Path, base_dir: Path) -> bool:
        """检查目标路径是否在允许的目录内"""
        try:
            target.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            return False
```

### 4.4 Skill Tool 创建

```python
# src/pure_agent_loop/skill/tool.py

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

    # 构建工具描述，包含所有可用 Skill 列表
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
        description_lines.append(f"  <skill>")
        description_lines.append(f"    <name>{skill.name}</name>")
        description_lines.append(f"    <description>{skill.description}</description>")
        if skill.scripts:
            description_lines.append(f"    <scripts>")
            for script in skill.scripts:
                description_lines.append(
                    f"      <script name=\"{script.name}\">{script.description}</script>"
                )
            description_lines.append(f"    </scripts>")
        description_lines.append(f"  </skill>")

    description_lines.append("</available_skills>")
    description = "\n".join(description_lines)

    async def execute_skill_action(
        action: str,
        name: str,
        script: str = "",
        args: dict[str, Any] | None = None,
        file_path: str = "",
    ) -> str:
        """执行 skill 工具操作"""
        if action == "load":
            skill_content = await registry.load(name)
            if not skill_content:
                available = [s.name for s in registry.get_all()]
                return f"错误：Skill '{name}' 不存在。可用 Skill：{available}"

            # 格式化输出
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

        elif action == "run":
            if not script:
                return "错误：action='run' 时必须指定 script 参数"
            return await registry.execute_script(name, script, args)

        elif action == "read_file":
            if not file_path:
                return "错误：action='read_file' 时必须指定 file_path 参数"
            return await registry.read_file(name, file_path)

        else:
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
                    "description": "操作类型：load（加载知识）、run（执行脚本）、read_file（读取资源）"
                },
                "name": {
                    "type": "string",
                    "description": "Skill 名称（从 available_skills 中选择）"
                },
                "script": {
                    "type": "string",
                    "description": "脚本文件名（action=run 时必需）"
                },
                "args": {
                    "type": "object",
                    "description": "脚本参数（action=run 时可选）"
                },
                "file_path": {
                    "type": "string",
                    "description": "相对于 Skill 目录的文件路径（action=read_file 时必需）"
                }
            },
            "required": ["action", "name"]
        },
        function=execute_skill_action,
        is_async=True,
    )
```

---

## 5. Agent 集成

### 5.1 Agent 参数扩展

```python
# src/pure_agent_loop/agent.py 修改

class Agent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_prompt: str = "",
        name: str = "智能助理",
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        thinking_level: ThinkingLevel = "off",
        emit_reasoning_events: bool = False,
        # ===== 新增参数 =====
        skills_dir: str | list[str] | None = None,
        **llm_kwargs: Any,
    ):
        # ... 现有初始化代码 ...

        # 初始化 Skill 系统
        self._skill_registry: SkillRegistry | None = None
        if skills_dir:
            from .skill.registry import SkillRegistry
            dirs = [Path(skills_dir)] if isinstance(skills_dir, str) else [Path(d) for d in skills_dir]
            self._skill_registry = SkillRegistry(dirs)

        self._skills_initialized = False

    async def _ensure_skills_initialized(self) -> None:
        """确保 Skill 系统已初始化"""
        if self._skills_initialized:
            return

        if self._skill_registry:
            await self._skill_registry.initialize()

            # 动态创建 skill 工具并注册
            if self._skill_registry.size > 0:
                from .skill.tool import create_skill_tool
                skill_tool = create_skill_tool(self._skill_registry)
                self._tool_registry.register(skill_tool)

        self._skills_initialized = True

    async def arun_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        # 确保 Skill 初始化
        await self._ensure_skills_initialized()

        # ... 现有代码 ...
```

---

## 6. 文件结构

### 6.1 新增文件

```
src/pure_agent_loop/
├── __init__.py              # 修改：新增导出
├── agent.py                 # 修改：新增 skills_dir 参数
├── skill/                   # 新增目录
│   ├── __init__.py
│   ├── types.py             # SkillInfo, SkillScript, SkillContent
│   ├── parser.py            # frontmatter 解析器
│   ├── scanner.py           # 目录扫描器
│   ├── registry.py          # SkillRegistry 核心
│   └── tool.py              # create_skill_tool()
├── ...

tests/
├── test_skill/              # 新增测试目录
│   ├── __init__.py
│   ├── test_parser.py       # 解析器测试
│   ├── test_scanner.py      # 扫描器测试
│   ├── test_registry.py     # Registry 测试
│   ├── test_security.py     # 安全机制测试
│   └── fixtures/
│       └── sample-skill/
│           ├── SKILL.md
│           ├── templates/
│           │   └── sample.txt
│           └── scripts/
│               └── sample.py

examples/
├── skills/                  # 示例 Skill 目录
│   └── web-search/
│       └── SKILL.md
├── exa_search_v2.py         # 使用 Skill 的优化版示例
```

---

## 7. 使用示例

### 7.1 优化后的 exa_search.py

```python
"""使用 Skill 系统的 Exa 搜索示例"""

import asyncio
import os
from pathlib import Path

from pure_agent_loop import Agent
from rich_renderer import RichRenderer

# 工具定义（保持不变）
from tools import exa_search, fetch_webpage, calculate


async def main():
    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL"),
        tools=[exa_search, fetch_webpage, calculate],
        skills_dir="./skills",  # 指向 skill 目录
        system_prompt=(
            "你是一个专业的助手。\n\n"
            "工作原则：\n"
            "1. 如果任务涉及网络搜索或信息检索，请先使用 skill 工具加载 'web-search' 技能获取详细指导\n"
            "2. 根据技能指导合理使用工具\n"
            "3. 注明信息来源"
        ),
    )

    query = "我想了解一下灵迹岛这款桌游的规则"
    print(f"\n🔍 查询: {query}\n")

    renderer = RichRenderer()
    async for event in agent.arun_stream(query):
        renderer.render(event)


if __name__ == "__main__":
    asyncio.run(main())
```

### 7.2 Skill 文件示例

**`examples/skills/web-search/SKILL.md`**:

```markdown
---
name: web-search
description: 当用户需要进行网络搜索、信息检索、获取实时数据时使用此技能
scripts: []
---

# 网络搜索技能指南

## 适用场景
- 用户询问需要实时信息的问题
- 需要查找网页内容或文档
- 搜索 API 返回结果不足时需要深入抓取

## 工具使用策略

### 1. exa_search
首选工具，用于搜索相关网页。

**使用时机**：
- 需要查找某个主题的相关信息
- 需要获取多个来源的信息

**注意**：
- 搜索结果包含标题、URL 和简短摘要
- 如果摘要信息不足，需要进一步使用 fetch_webpage

### 2. fetch_webpage
用于获取网页的详细内容。

**使用时机**：
- exa_search 结果的摘要不够详细
- 需要查看特定网页的完整内容
- 需要引用网页中的具体信息

**注意**：
- 优先选择官方/权威来源
- 避免过度抓取，通常 2-3 个网页足够

## 工作流程

```
1. 用户提问
     ↓
2. 使用 exa_search 搜索关键词
     ↓
3. 分析搜索结果，筛选最相关的 3-5 个
     ↓
4. 判断摘要是否足够回答问题
   ├─ 是 → 直接综合回答
   └─ 否 → 使用 fetch_webpage 获取详情
     ↓
5. 综合所有信息回答，注明来源
```

## 注意事项
- 始终注明信息来源 URL
- 区分事实与观点
- 如信息有时效性，注明检索时间
```

---

## 8. 测试计划

### 8.1 单元测试

| 测试文件 | 测试内容 |
|---------|---------|
| `test_parser.py` | frontmatter 解析、必需字段验证、YAML 错误处理 |
| `test_scanner.py` | 目录扫描、多目录支持、去重逻辑 |
| `test_registry.py` | 初始化、get/get_all、load、重复名称处理 |
| `test_security.py` | 路径越界检测、未声明脚本拒绝、白名单校验 |

### 8.2 集成测试

- Agent 初始化时自动加载 Skill
- skill 工具正确生成描述
- load/run/read_file 三种 action 正确执行
- 与现有 Todo 系统协同工作

---

## 9. 实施步骤

1. **创建 skill 模块基础结构**
   - 创建 `src/pure_agent_loop/skill/` 目录
   - 实现 types.py、parser.py、scanner.py

2. **实现 SkillRegistry**
   - 实现核心逻辑
   - 实现安全机制

3. **实现 skill 工具**
   - 实现 create_skill_tool()
   - 实现三种 action 逻辑

4. **集成到 Agent**
   - 修改 Agent.__init__() 添加 skills_dir 参数
   - 添加懒初始化逻辑

5. **编写测试**
   - 创建测试 fixtures
   - 编写单元测试和集成测试

6. **更新示例和文档**
   - 创建 examples/skills/ 目录
   - 更新 exa_search.py 示例
   - 更新 README.md

---

## 10. 附录：安全考虑清单

- [x] 脚本必须在 SKILL.md 中显式声明
- [x] 使用 Path.resolve() 规范化路径
- [x] 检查路径是否在白名单目录内
- [x] 脚本执行超时限制（60秒）
- [x] 在 skill 目录下执行脚本（隔离工作目录）
- [ ] （可选）限制可执行脚本的类型（仅 .py）
- [ ] （可选）添加脚本执行审计日志
