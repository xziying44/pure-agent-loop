# Skill 动态知识注入系统 - 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 框架添加 Skill 系统，支持按需加载专业领域知识、执行脚本和读取资源文件。

**Architecture:** 采用两阶段加载策略（启动时索引 frontmatter + 按需加载完整内容），通过白名单路径机制确保脚本执行和文件读取安全。Skill 作为内置工具集成到 Agent，LLM 自主决定何时调用。

**Tech Stack:** Python 3.10+, PyYAML, dataclasses, subprocess, pytest, pytest-asyncio

---

## 项目约定

- **虚拟环境**: 执行任何 Python 命令前，确认已激活 `venv`
- **测试命令**: `pytest tests/test_skill/ -v`
- **代码规范**: 所有注释和 docstring 使用中文
- **类型注解**: 使用 Python 3.10+ 语法（`X | None`，`list[T]`）

---

## Task 1: 创建 skill 模块目录结构

**Files:**
- Create: `src/pure_agent_loop/skill/__init__.py`
- Create: `src/pure_agent_loop/skill/types.py`
- Create: `tests/test_skill/__init__.py`

**Step 1: 创建 skill 模块目录**

```bash
mkdir -p src/pure_agent_loop/skill
mkdir -p tests/test_skill
```

**Step 2: 创建 skill 模块 __init__.py**

```python
# src/pure_agent_loop/skill/__init__.py
"""Skill 动态知识注入系统

支持按需加载专业领域知识、执行脚本和读取资源文件。
"""

from .types import SkillInfo, SkillScript, SkillContent

__all__ = [
    "SkillInfo",
    "SkillScript",
    "SkillContent",
]
```

**Step 3: 创建数据类型定义 types.py**

```python
# src/pure_agent_loop/skill/types.py
"""Skill 系统数据类型定义"""

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
```

**Step 4: 创建测试模块 __init__.py**

```python
# tests/test_skill/__init__.py
"""Skill 系统测试模块"""
```

**Step 5: 验证文件创建成功**

Run: `ls -la src/pure_agent_loop/skill/`
Expected: 显示 `__init__.py` 和 `types.py`

Run: `python -c "from pure_agent_loop.skill import SkillInfo, SkillScript, SkillContent; print('导入成功')"`
Expected: `导入成功`

**Step 6: Commit**

```bash
git add src/pure_agent_loop/skill/ tests/test_skill/
git commit -m "feat(skill): 添加 skill 模块基础结构和数据类型定义"
```

---

## Task 2: 实现 Frontmatter 解析器

**Files:**
- Create: `src/pure_agent_loop/skill/parser.py`
- Create: `tests/test_skill/test_parser.py`
- Create: `tests/test_skill/fixtures/valid-skill/SKILL.md`
- Create: `tests/test_skill/fixtures/invalid-skill/SKILL.md`

**Step 1: 创建测试 fixtures 目录**

```bash
mkdir -p tests/test_skill/fixtures/valid-skill
mkdir -p tests/test_skill/fixtures/invalid-skill
```

**Step 2: 创建有效的测试 SKILL.md**

```markdown
---
name: test-skill
description: 这是一个用于测试的技能
scripts:
  - name: hello.py
    description: 打印问候语
    args_schema:
      name: { type: string, required: true }
---

# 测试技能

这是技能的正文内容。

## 使用说明

这里是详细的使用说明。
```

保存到: `tests/test_skill/fixtures/valid-skill/SKILL.md`

**Step 3: 创建无效的测试 SKILL.md（缺少必需字段）**

```markdown
---
name: incomplete-skill
---

# 不完整的技能

缺少 description 字段。
```

保存到: `tests/test_skill/fixtures/invalid-skill/SKILL.md`

**Step 4: 编写解析器测试**

```python
# tests/test_skill/test_parser.py
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
        assert info.base_dir == skill_file.parent

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

    def test_parse_frontmatter_no_scripts(self):
        """测试没有脚本声明的 SKILL.md"""
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""---
name: no-scripts-skill
description: 没有脚本的技能
---

# 内容
""")
            temp_path = Path(f.name)

        try:
            info = SkillParser.parse_frontmatter(temp_path)
            assert info is not None
            assert info.name == "no-scripts-skill"
            assert info.scripts == []
        finally:
            temp_path.unlink()
```

**Step 5: 运行测试确认失败**

Run: `pytest tests/test_skill/test_parser.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pure_agent_loop.skill.parser'"

**Step 6: 实现解析器**

```python
# src/pure_agent_loop/skill/parser.py
"""SKILL.md 文件解析器"""

from pathlib import Path
from typing import Any

import yaml

from .types import SkillInfo, SkillScript


class SkillParser:
    """SKILL.md 文件解析器

    支持解析 YAML frontmatter 和 Markdown 正文。
    """

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
        scripts: list[SkillScript] = []
        for script_data in metadata.get("scripts", []):
            scripts.append(SkillScript(
                name=script_data.get("name", ""),
                description=script_data.get("description", ""),
                args_schema=script_data.get("args_schema", {}),
            ))

        base_dir = file_path.parent.resolve()

        return SkillInfo(
            name=metadata["name"],
            description=metadata["description"],
            location=file_path.resolve(),
            base_dir=base_dir,
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
        except Exception as e:
            print(f"[SkillParser] 读取文件失败: {file_path} - {e}")
            return None, ""

        # 解析 frontmatter
        info = cls.parse_frontmatter(file_path)
        if not info:
            return None, ""

        # 提取正文（第二个 --- 之后的内容）
        end_index = raw.find(cls.DELIMITER, 3)
        body_content = raw[end_index + 3:].strip()

        return info, body_content
```

**Step 7: 运行测试确认通过**

Run: `pytest tests/test_skill/test_parser.py -v`
Expected: All tests PASS

**Step 8: 更新 __init__.py 导出**

```python
# src/pure_agent_loop/skill/__init__.py
"""Skill 动态知识注入系统

支持按需加载专业领域知识、执行脚本和读取资源文件。
"""

from .types import SkillInfo, SkillScript, SkillContent
from .parser import SkillParser

__all__ = [
    "SkillInfo",
    "SkillScript",
    "SkillContent",
    "SkillParser",
]
```

**Step 9: Commit**

```bash
git add src/pure_agent_loop/skill/parser.py src/pure_agent_loop/skill/__init__.py tests/test_skill/
git commit -m "feat(skill): 实现 Frontmatter 解析器及测试"
```

---

## Task 3: 实现目录扫描器

**Files:**
- Create: `src/pure_agent_loop/skill/scanner.py`
- Create: `tests/test_skill/test_scanner.py`
- Create: `tests/test_skill/fixtures/skill-dir/skill-a/SKILL.md`
- Create: `tests/test_skill/fixtures/skill-dir/skill-b/SKILL.md`

**Step 1: 创建测试 fixtures**

```bash
mkdir -p tests/test_skill/fixtures/skill-dir/skill-a
mkdir -p tests/test_skill/fixtures/skill-dir/skill-b
```

**Step 2: 创建 skill-a/SKILL.md**

```markdown
---
name: skill-a
description: 技能 A 的描述
---

# 技能 A
```

保存到: `tests/test_skill/fixtures/skill-dir/skill-a/SKILL.md`

**Step 3: 创建 skill-b/SKILL.md**

```markdown
---
name: skill-b
description: 技能 B 的描述
---

# 技能 B
```

保存到: `tests/test_skill/fixtures/skill-dir/skill-b/SKILL.md`

**Step 4: 编写扫描器测试**

```python
# tests/test_skill/test_scanner.py
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

    def test_scan_empty_directory(self):
        """测试扫描空目录返回空列表"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = SkillScanner([Path(tmpdir)])
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
```

**Step 5: 运行测试确认失败**

Run: `pytest tests/test_skill/test_scanner.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pure_agent_loop.skill.scanner'"

**Step 6: 实现扫描器**

```python
# src/pure_agent_loop/skill/scanner.py
"""Skill 目录扫描器"""

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
            SKILL.md 文件的绝对路径列表（已去重）
        """
        results: set[Path] = set()

        for base_dir in self.directories:
            if not base_dir.exists():
                continue

            # 扫描直接子目录下的 SKILL.md
            for skill_file in base_dir.glob(f"*/{self.SKILL_FILE}"):
                results.add(skill_file.resolve())

        return list(results)
```

**Step 7: 运行测试确认通过**

Run: `pytest tests/test_skill/test_scanner.py -v`
Expected: All tests PASS

**Step 8: 更新 __init__.py 导出**

```python
# src/pure_agent_loop/skill/__init__.py
"""Skill 动态知识注入系统

支持按需加载专业领域知识、执行脚本和读取资源文件。
"""

from .types import SkillInfo, SkillScript, SkillContent
from .parser import SkillParser
from .scanner import SkillScanner

__all__ = [
    "SkillInfo",
    "SkillScript",
    "SkillContent",
    "SkillParser",
    "SkillScanner",
]
```

**Step 9: Commit**

```bash
git add src/pure_agent_loop/skill/scanner.py src/pure_agent_loop/skill/__init__.py tests/test_skill/
git commit -m "feat(skill): 实现目录扫描器及测试"
```

---

## Task 4: 实现 SkillRegistry 核心（索引和加载）

**Files:**
- Create: `src/pure_agent_loop/skill/registry.py`
- Create: `tests/test_skill/test_registry.py`

**Step 1: 编写 Registry 测试（初始化和加载）**

```python
# tests/test_skill/test_registry.py
"""SkillRegistry 测试"""

from pathlib import Path
import pytest

from pure_agent_loop.skill.registry import SkillRegistry


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillRegistry:
    """SkillRegistry 测试类"""

    @pytest.fixture
    def registry(self):
        """创建 Registry 实例"""
        skill_dir = FIXTURES_DIR / "skill-dir"
        return SkillRegistry([skill_dir])

    async def test_initialize(self, registry):
        """测试初始化加载索引"""
        await registry.initialize()

        assert registry.initialized
        assert registry.size == 2

    async def test_initialize_idempotent(self, registry):
        """测试多次初始化是幂等的"""
        await registry.initialize()
        await registry.initialize()

        assert registry.size == 2

    async def test_get_existing_skill(self, registry):
        """测试获取已存在的 Skill"""
        await registry.initialize()

        info = registry.get("skill-a")
        assert info is not None
        assert info.name == "skill-a"

    async def test_get_nonexistent_skill(self, registry):
        """测试获取不存在的 Skill 返回 None"""
        await registry.initialize()

        info = registry.get("nonexistent")
        assert info is None

    async def test_get_all(self, registry):
        """测试获取所有 Skill"""
        await registry.initialize()

        skills = registry.get_all()
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names

    async def test_load_skill_content(self, registry):
        """测试加载 Skill 完整内容"""
        await registry.initialize()

        content = await registry.load("skill-a")
        assert content is not None
        assert content.name == "skill-a"
        assert "技能 A" in content.content

    async def test_load_nonexistent_skill(self, registry):
        """测试加载不存在的 Skill 返回 None"""
        await registry.initialize()

        content = await registry.load("nonexistent")
        assert content is None
```

**Step 2: 运行测试确认失败**

Run: `pytest tests/test_skill/test_registry.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pure_agent_loop.skill.registry'"

**Step 3: 实现 Registry（第一部分：索引和加载）**

```python
# src/pure_agent_loop/skill/registry.py
"""Skill 注册表"""

import asyncio
from pathlib import Path
from typing import Any

from .types import SkillInfo, SkillContent
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
        """是否已初始化"""
        return self._initialized

    @property
    def size(self) -> int:
        """已注册 Skill 数量"""
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

                self._skills[info.name] = info
                print(f"[SkillRegistry] 已注册: {info.name}")

            self._initialized = True

    def get(self, name: str) -> SkillInfo | None:
        """获取单个 Skill 信息

        Args:
            name: Skill 名称

        Returns:
            SkillInfo 或 None
        """
        return self._skills.get(name)

    def get_all(self) -> list[SkillInfo]:
        """获取所有 Skill 信息

        Returns:
            所有已注册 Skill 的列表
        """
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

    def _is_path_safe(self, target: Path, base_dir: Path) -> bool:
        """检查目标路径是否在允许的目录内

        Args:
            target: 目标路径
            base_dir: 基准目录

        Returns:
            是否安全
        """
        try:
            target.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            return False
```

**Step 4: 运行测试确认通过**

Run: `pytest tests/test_skill/test_registry.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/pure_agent_loop/skill/registry.py tests/test_skill/test_registry.py
git commit -m "feat(skill): 实现 SkillRegistry 索引和加载功能"
```

---

## Task 5: 实现 SkillRegistry 安全功能（read_file 和 execute_script）

**Files:**
- Modify: `src/pure_agent_loop/skill/registry.py`
- Create: `tests/test_skill/test_security.py`
- Create: `tests/test_skill/fixtures/script-skill/SKILL.md`
- Create: `tests/test_skill/fixtures/script-skill/scripts/hello.py`
- Create: `tests/test_skill/fixtures/script-skill/templates/sample.txt`

**Step 1: 创建带脚本的测试 Skill**

```bash
mkdir -p tests/test_skill/fixtures/script-skill/scripts
mkdir -p tests/test_skill/fixtures/script-skill/templates
```

**Step 2: 创建 script-skill/SKILL.md**

```markdown
---
name: script-skill
description: 用于测试脚本执行的技能
scripts:
  - name: hello.py
    description: 打印问候语
    args_schema:
      name: { type: string }
---

# 脚本测试技能

## 资源文件
- 模板: `templates/sample.txt`
```

保存到: `tests/test_skill/fixtures/script-skill/SKILL.md`

**Step 3: 创建测试脚本**

```python
#!/usr/bin/env python
# tests/test_skill/fixtures/script-skill/scripts/hello.py
"""测试脚本"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="World")
args = parser.parse_args()

print(f"Hello, {args.name}!")
```

保存到: `tests/test_skill/fixtures/script-skill/scripts/hello.py`

**Step 4: 创建模板文件**

```text
这是一个示例模板文件。
用于测试 read_file 功能。
```

保存到: `tests/test_skill/fixtures/script-skill/templates/sample.txt`

**Step 5: 编写安全功能测试**

```python
# tests/test_skill/test_security.py
"""Skill 安全机制测试"""

from pathlib import Path
import pytest

from pure_agent_loop.skill.registry import SkillRegistry


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillSecurity:
    """安全机制测试类"""

    @pytest.fixture
    async def registry(self):
        """创建包含脚本 Skill 的 Registry"""
        skill_dir = FIXTURES_DIR
        reg = SkillRegistry([skill_dir])
        await reg.initialize()
        return reg

    # ===== read_file 测试 =====

    async def test_read_file_success(self, registry):
        """测试成功读取资源文件"""
        result = await registry.read_file("script-skill", "templates/sample.txt")

        assert "示例模板文件" in result
        assert "错误" not in result

    async def test_read_file_nonexistent_skill(self, registry):
        """测试读取不存在 Skill 的文件"""
        result = await registry.read_file("nonexistent", "file.txt")

        assert "错误" in result
        assert "不存在" in result

    async def test_read_file_nonexistent_file(self, registry):
        """测试读取不存在的文件"""
        result = await registry.read_file("script-skill", "nonexistent.txt")

        assert "错误" in result
        assert "不存在" in result

    async def test_read_file_path_traversal_blocked(self, registry):
        """测试路径遍历攻击被阻止"""
        result = await registry.read_file("script-skill", "../valid-skill/SKILL.md")

        assert "错误" in result
        assert "越界" in result

    async def test_read_file_absolute_path_blocked(self, registry):
        """测试绝对路径被正确处理"""
        # 即使传入绝对路径，也会被拼接到 base_dir
        result = await registry.read_file("script-skill", "/etc/passwd")

        # 应该找不到文件或路径越界
        assert "错误" in result

    # ===== execute_script 测试 =====

    async def test_execute_script_success(self, registry):
        """测试成功执行脚本"""
        result = await registry.execute_script(
            "script-skill", "hello.py", {"name": "Test"}
        )

        assert "Hello, Test!" in result

    async def test_execute_script_default_args(self, registry):
        """测试使用默认参数执行脚本"""
        result = await registry.execute_script("script-skill", "hello.py", None)

        assert "Hello, World!" in result

    async def test_execute_script_nonexistent_skill(self, registry):
        """测试执行不存在 Skill 的脚本"""
        result = await registry.execute_script("nonexistent", "hello.py", None)

        assert "错误" in result
        assert "不存在" in result

    async def test_execute_script_undeclared_script(self, registry):
        """测试执行未声明的脚本被拒绝"""
        result = await registry.execute_script("script-skill", "undeclared.py", None)

        assert "错误" in result
        assert "未在" in result or "未声明" in result

    async def test_execute_script_path_traversal_blocked(self, registry):
        """测试脚本路径遍历被阻止"""
        # 即使脚本名包含路径遍历，也应该被阻止
        result = await registry.execute_script(
            "script-skill", "../other/script.py", None
        )

        assert "错误" in result
```

**Step 6: 运行测试确认失败**

Run: `pytest tests/test_skill/test_security.py -v`
Expected: FAIL with "AttributeError: 'SkillRegistry' object has no attribute 'read_file'"

**Step 7: 实现 read_file 和 execute_script**

在 `registry.py` 中添加以下方法：

```python
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
        import subprocess

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

            return output.strip() or "(脚本执行完成，无输出)"

        except subprocess.TimeoutExpired:
            return "错误：脚本执行超时（60秒）"
        except Exception as e:
            return f"错误：脚本执行失败 - {e}"
```

**Step 8: 运行测试确认通过**

Run: `pytest tests/test_skill/test_security.py -v`
Expected: All tests PASS

**Step 9: 运行全部 skill 测试**

Run: `pytest tests/test_skill/ -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/pure_agent_loop/skill/registry.py tests/test_skill/
git commit -m "feat(skill): 实现 read_file 和 execute_script 安全功能"
```

---

## Task 6: 实现 skill 工具

**Files:**
- Create: `src/pure_agent_loop/skill/tool.py`
- Create: `tests/test_skill/test_tool.py`

**Step 1: 编写工具测试**

```python
# tests/test_skill/test_tool.py
"""skill 工具测试"""

from pathlib import Path
import pytest

from pure_agent_loop.skill.registry import SkillRegistry
from pure_agent_loop.skill.tool import create_skill_tool


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillTool:
    """skill 工具测试类"""

    @pytest.fixture
    async def registry(self):
        """创建 Registry"""
        reg = SkillRegistry([FIXTURES_DIR])
        await reg.initialize()
        return reg

    @pytest.fixture
    def skill_tool(self, registry):
        """创建 skill 工具"""
        return create_skill_tool(registry)

    def test_tool_name(self, skill_tool):
        """测试工具名称"""
        assert skill_tool.name == "skill"

    def test_tool_description_contains_skills(self, skill_tool):
        """测试工具描述包含可用 Skill 列表"""
        desc = skill_tool.description
        assert "skill-a" in desc
        assert "skill-b" in desc
        assert "script-skill" in desc

    def test_tool_parameters(self, skill_tool):
        """测试工具参数定义"""
        params = skill_tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "name" in params["properties"]
        assert params["properties"]["action"]["enum"] == ["load", "run", "read_file"]

    async def test_execute_load_action(self, skill_tool):
        """测试 load 操作"""
        result = await skill_tool.execute({
            "action": "load",
            "name": "skill-a"
        })

        assert "skill-a" in result
        assert "技能 A" in result

    async def test_execute_load_nonexistent(self, skill_tool):
        """测试 load 不存在的 Skill"""
        result = await skill_tool.execute({
            "action": "load",
            "name": "nonexistent"
        })

        assert "错误" in result

    async def test_execute_read_file_action(self, skill_tool):
        """测试 read_file 操作"""
        result = await skill_tool.execute({
            "action": "read_file",
            "name": "script-skill",
            "file_path": "templates/sample.txt"
        })

        assert "示例模板文件" in result

    async def test_execute_run_action(self, skill_tool):
        """测试 run 操作"""
        result = await skill_tool.execute({
            "action": "run",
            "name": "script-skill",
            "script": "hello.py",
            "args": {"name": "Claude"}
        })

        assert "Hello, Claude!" in result

    async def test_execute_unknown_action(self, skill_tool):
        """测试未知操作"""
        result = await skill_tool.execute({
            "action": "unknown",
            "name": "skill-a"
        })

        assert "错误" in result
        assert "未知" in result
```

**Step 2: 运行测试确认失败**

Run: `pytest tests/test_skill/test_tool.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pure_agent_loop.skill.tool'"

**Step 3: 实现 skill 工具**

```python
# src/pure_agent_loop/skill/tool.py
"""skill 工具创建"""

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

    async def execute_skill_action(**kwargs: Any) -> str:
        """执行 skill 工具操作"""
        action = kwargs.get("action", "")
        name = kwargs.get("name", "")
        script = kwargs.get("script", "")
        args = kwargs.get("args")
        file_path = kwargs.get("file_path", "")

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

**Step 4: 运行测试确认通过**

Run: `pytest tests/test_skill/test_tool.py -v`
Expected: All tests PASS

**Step 5: 更新 __init__.py 导出**

```python
# src/pure_agent_loop/skill/__init__.py
"""Skill 动态知识注入系统

支持按需加载专业领域知识、执行脚本和读取资源文件。
"""

from .types import SkillInfo, SkillScript, SkillContent
from .parser import SkillParser
from .scanner import SkillScanner
from .registry import SkillRegistry
from .tool import create_skill_tool

__all__ = [
    "SkillInfo",
    "SkillScript",
    "SkillContent",
    "SkillParser",
    "SkillScanner",
    "SkillRegistry",
    "create_skill_tool",
]
```

**Step 6: Commit**

```bash
git add src/pure_agent_loop/skill/ tests/test_skill/test_tool.py
git commit -m "feat(skill): 实现 skill 工具"
```

---

## Task 7: 集成到 Agent

**Files:**
- Modify: `src/pure_agent_loop/agent.py`
- Modify: `src/pure_agent_loop/__init__.py`
- Create: `tests/test_skill/test_agent_integration.py`

**Step 1: 编写集成测试**

```python
# tests/test_skill/test_agent_integration.py
"""Agent Skill 集成测试"""

from pathlib import Path
import pytest

from pure_agent_loop import Agent
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class MockLLM(BaseLLMClient):
    """Mock LLM 客户端"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


class TestAgentSkillIntegration:
    """Agent Skill 集成测试"""

    async def test_agent_with_skills_dir(self):
        """测试 Agent 使用 skills_dir 参数"""
        # 创建 Mock LLM 响应序列
        responses = [
            # 第一次调用：LLM 决定调用 skill 工具
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="skill", arguments={"action": "load", "name": "skill-a"})
                ],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            # 第二次调用：LLM 给出最终回答
            LLMResponse(
                content="根据技能 A 的指导，任务已完成。",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(
            llm=mock_llm,
            skills_dir=str(FIXTURES_DIR / "skill-dir"),
        )

        result = await agent.arun("请使用技能 A 完成任务")

        assert result.stop_reason == "completed"
        assert "任务已完成" in result.content

    async def test_agent_skill_tool_registered(self):
        """测试 skill 工具被正确注册"""
        responses = [
            LLMResponse(
                content="完成",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(
            llm=mock_llm,
            skills_dir=str(FIXTURES_DIR / "skill-dir"),
        )

        # 触发初始化
        await agent.arun("test")

        # 检查 skill 工具是否注册
        skill_tool = agent._tool_registry.get("skill")
        assert skill_tool is not None
        assert "skill-a" in skill_tool.description
        assert "skill-b" in skill_tool.description

    async def test_agent_without_skills_dir(self):
        """测试不使用 skills_dir 时正常工作"""
        responses = [
            LLMResponse(
                content="完成",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        mock_llm = MockLLM(responses)
        agent = Agent(llm=mock_llm)  # 不传 skills_dir

        result = await agent.arun("test")

        assert result.stop_reason == "completed"
        # skill 工具不应该注册
        skill_tool = agent._tool_registry.get("skill")
        assert skill_tool is None
```

**Step 2: 运行测试确认失败**

Run: `pytest tests/test_skill/test_agent_integration.py -v`
Expected: FAIL with "Agent() got an unexpected keyword argument 'skills_dir'"

**Step 3: 修改 Agent 添加 skills_dir 参数**

修改 `src/pure_agent_loop/agent.py`：

```python
# 在文件顶部添加导入
from pathlib import Path

# ... 其他导入保持不变 ...

class Agent:
    """Agent 入口类

    使用 pure-agent-loop 的唯一入口。

    使用方式 1 - 通过 model 参数（内置 OpenAI 兼容客户端）:
        agent = Agent(model="deepseek-chat", api_key="sk-xxx", base_url="...")

    使用方式 2 - 通过自定义 LLM 客户端:
        agent = Agent(llm=MyCustomClient())

    Args:
        model: 模型名称（使用内置客户端时）
        api_key: API 密钥（默认读取环境变量）
        base_url: API 基础地址
        llm: 自定义 LLM 客户端实例（与 model 二选一）
        tools: 工具列表（@tool 装饰器或字典格式）
        system_prompt: 系统提示
        limits: 终止条件配置
        retry: 重试配置
        temperature: 温度参数
        thinking_level: 思考深度（off/low/medium/high），默认 off
        emit_reasoning_events: 是否推送 REASONING 事件，默认 False
        skills_dir: Skill 目录路径（支持字符串或列表）
        **llm_kwargs: 透传给 LLM 调用的额外参数
    """

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
        skills_dir: str | list[str] | None = None,
        **llm_kwargs: Any,
    ):
        # 保存思考模式配置
        self._thinking_level = thinking_level
        self._emit_reasoning_events = emit_reasoning_events

        # 构建 LLM 客户端
        if llm is not None:
            self._llm = llm
        else:
            self._llm = OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                thinking_level=thinking_level,
            )

        # 创建 TodoStore 和内置工具
        self._todo_store = TodoStore()
        self._name = name

        # 注册工具（内置 + 用户）
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(create_todo_tool(self._todo_store))
        if tools:
            self._tool_registry.register_many(tools)

        # 构建完整系统提示词
        self._system_prompt = build_system_prompt(
            name=name,
            user_prompt=system_prompt,
        )
        self._limits = limits or LoopLimits()
        self._retry = retry or RetryConfig()
        self._llm_kwargs: dict[str, Any] = {"temperature": temperature, **llm_kwargs}

        # 初始化 Skill 系统
        self._skill_registry = None
        self._skills_initialized = False
        if skills_dir:
            from .skill.registry import SkillRegistry
            if isinstance(skills_dir, str):
                dirs = [Path(skills_dir)]
            else:
                dirs = [Path(d) for d in skills_dir]
            self._skill_registry = SkillRegistry(dirs)

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

    # ... _create_loop 方法保持不变 ...

    async def arun_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """异步流式执行

        Args:
            task: 任务描述
            messages: 初始消息历史（多轮对话续接）

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 确保 Skill 系统已初始化
        await self._ensure_skills_initialized()

        loop = self._create_loop()
        async for event in loop.run(
            task=task,
            system_prompt=self._system_prompt,
            messages=messages,
        ):
            yield event

    # ... 其他方法保持不变 ...
```

**Step 4: 运行测试确认通过**

Run: `pytest tests/test_skill/test_agent_integration.py -v`
Expected: All tests PASS

**Step 5: 更新主模块 __init__.py 导出**

```python
# src/pure_agent_loop/__init__.py
# 在现有导出后添加
from .skill import SkillRegistry, SkillInfo, create_skill_tool

# 更新 __all__ 列表
__all__ = [
    # ... 现有导出 ...
    # Skill 系统
    "SkillRegistry",
    "SkillInfo",
    "create_skill_tool",
]
```

**Step 6: 运行全部测试**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/pure_agent_loop/agent.py src/pure_agent_loop/__init__.py tests/test_skill/test_agent_integration.py
git commit -m "feat(skill): 将 Skill 系统集成到 Agent"
```

---

## Task 8: 创建示例

**Files:**
- Create: `examples/skills/web-search/SKILL.md`
- Create: `examples/exa_search_with_skill.py`

**Step 1: 创建示例 Skill 目录**

```bash
mkdir -p examples/skills/web-search
```

**Step 2: 创建 web-search Skill**

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

保存到: `examples/skills/web-search/SKILL.md`

**Step 3: 创建使用 Skill 的示例**

```python
# examples/exa_search_with_skill.py
"""使用 Skill 系统的 Exa 搜索示例

这个示例展示了如何使用 Skill 系统来指导智能体的工具使用策略。
与原版 exa_search.py 相比，系统提示词更简洁，工具使用策略通过 Skill 动态加载。
"""

import asyncio
import os
import re
from pathlib import Path

import html2text
import requests
from dotenv import load_dotenv

from pure_agent_loop import Agent, tool, ThinkingLevel

# 尝试导入 rich_renderer（如果存在）
try:
    from rich_renderer import RichRenderer
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")

# Exa API 配置
EXA_API_URL = "https://api.exa.ai/search"
EXA_API_KEY = os.getenv("EXA_API_KEY", "")


@tool
def exa_search(query: str, num_results: int = 5) -> str:
    """使用 Exa AI 搜索网页内容

    Args:
        query: 搜索查询词
        num_results: 返回结果数量，默认5条
    """
    if not EXA_API_KEY:
        return "错误: 未配置 EXA_API_KEY 环境变量"

    try:
        response = requests.post(
            EXA_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": EXA_API_KEY,
            },
            json={
                "query": query,
                "type": "auto",
                "numResults": num_results,
                "contents": {"text": True},
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return f"未找到关于 '{query}' 的搜索结果"

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            text = result.get("text", "")[:500]
            if len(result.get("text", "")) > 500:
                text += "..."
            formatted.append(f"[{i}] {title}\n    URL: {url}\n    内容: {text}\n")

        return "\n".join(formatted)

    except requests.exceptions.Timeout:
        return "错误: 搜索请求超时"
    except requests.exceptions.RequestException as e:
        return f"错误: 搜索请求失败 - {e}"
    except Exception as e:
        return f"错误: {e}"


@tool
def fetch_webpage(url: str, max_length: int = 8000) -> str:
    """访问指定 URL 并将网页内容转换为 Markdown 格式返回

    Args:
        url: 要访问的网页 URL（必须是完整的 http/https 地址）
        max_length: 返回内容的最大字符数，默认 8000
    """
    if not url.startswith(("http://", "https://")):
        return "错误: URL 必须以 http:// 或 https:// 开头"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        html_content = response.text

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0

        markdown_content = converter.handle(html_content)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "\n\n... [内容已截断]"

        return f"# 网页内容 ({url})\n\n{markdown_content}"

    except Exception as e:
        return f"错误: {e}"


async def main():
    if not os.getenv("API_KEY"):
        print("错误: 请在 .env 文件中配置 API_KEY")
        return
    if not EXA_API_KEY:
        print("错误: 请在 .env 文件中配置 EXA_API_KEY")
        return

    thinking_level: ThinkingLevel = os.getenv("THINKING_LEVEL", "off")  # type: ignore

    # 使用 Skill 系统的 Agent
    # 注意：系统提示词更简洁，工具使用策略通过 Skill 动态加载
    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[exa_search, fetch_webpage],
        skills_dir=str(Path(__file__).parent / "skills"),  # 指向 skills 目录
        system_prompt=(
            "你是一个专业的助手。\n\n"
            "工作原则：\n"
            "1. 如果任务涉及网络搜索或信息检索，请先使用 skill 工具加载 'web-search' 技能获取详细指导\n"
            "2. 根据技能指导合理使用工具\n"
            "3. 注明信息来源"
        ),
        thinking_level=thinking_level,
        emit_reasoning_events=thinking_level != "off",
    )

    query = "我想了解一下灵迹岛这款桌游的规则"
    print(f"\n🔍 查询: {query}\n")
    print("=" * 60)

    if HAS_RICH:
        renderer = RichRenderer(
            max_thought_lines=3,
            max_result_chars=150,
        )
        async for event in agent.arun_stream(query):
            renderer.render(event)
    else:
        async for event in agent.arun_stream(query):
            print(f"[{event.type.value}] {event.data}")

    print("\n" + "=" * 60)
    print("✅ 执行完成")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 4: 验证示例可以运行**

Run: `cd examples && python -c "from exa_search_with_skill import *; print('导入成功')"`
Expected: `导入成功`

**Step 5: Commit**

```bash
git add examples/skills/ examples/exa_search_with_skill.py
git commit -m "docs(examples): 添加 Skill 系统使用示例"
```

---

## Task 9: 最终验证和清理

**Step 1: 运行全部测试**

Run: `pytest tests/ -v --cov=pure_agent_loop --cov-report=term-missing`
Expected: All tests PASS, coverage >= 80%

**Step 2: 检查代码格式（如果有 linter）**

Run: `python -m py_compile src/pure_agent_loop/skill/*.py`
Expected: No syntax errors

**Step 3: 验证导入正常**

Run: `python -c "from pure_agent_loop import Agent, SkillRegistry, create_skill_tool; print('所有导入成功')"`
Expected: `所有导入成功`

**Step 4: 最终 Commit**

```bash
git add .
git commit -m "chore: Skill 系统实施完成"
```

---

## 完成检查清单

- [ ] Task 1: 创建 skill 模块目录结构
- [ ] Task 2: 实现 Frontmatter 解析器
- [ ] Task 3: 实现目录扫描器
- [ ] Task 4: 实现 SkillRegistry 核心（索引和加载）
- [ ] Task 5: 实现 SkillRegistry 安全功能（read_file 和 execute_script）
- [ ] Task 6: 实现 skill 工具
- [ ] Task 7: 集成到 Agent
- [ ] Task 8: 创建示例
- [ ] Task 9: 最终验证和清理
