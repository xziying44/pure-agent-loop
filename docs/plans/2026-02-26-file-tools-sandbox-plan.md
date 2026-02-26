# 文件工具 + 沙箱系统实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 添加 5 个内置文件工具（file_read、file_search、file_grep、file_edit、file_write）和沙箱权限系统。

**Architecture:** 新增 `sandbox.py`（Sandbox 配置 + SandboxGuard 验证器）和 `file_tools.py`（5 个工具 + 工厂函数）。Agent 新增 `sandbox` 参数，构造时自动创建并注册文件工具。保持 `builtin_tools.py` 不变。

**Tech Stack:** Python 3.10+ 标准库（pathlib, re, fnmatch, difflib），零外部依赖。

---

## Task 1: 创建 Sandbox 配置类和 SandboxGuard 验证器

**Files:**
- Create: `src/pure_agent_loop/sandbox.py`
- Create: `tests/test_sandbox.py`
- Modify: `src/pure_agent_loop/errors.py:38-51` (添加 SandboxViolationError)

**Step 1: 在 errors.py 中添加 SandboxViolationError 失败测试**

在 `tests/test_errors.py` 末尾追加：

```python
class TestSandboxViolationError:
    """沙箱违规异常测试"""

    def test_inherits_base(self):
        from pure_agent_loop.errors import SandboxViolationError, PureAgentLoopError
        err = SandboxViolationError("/etc/passwd", "read", ["/workspace"])
        assert isinstance(err, PureAgentLoopError)

    def test_message_contains_path(self):
        from pure_agent_loop.errors import SandboxViolationError
        err = SandboxViolationError("/etc/passwd", "read", ["/workspace"])
        assert "/etc/passwd" in str(err)
        assert "read" in str(err)
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_errors.py::TestSandboxViolationError -v`
Expected: FAIL (ImportError: cannot import name 'SandboxViolationError')

**Step 3: 在 errors.py 中实现 SandboxViolationError**

在 `src/pure_agent_loop/errors.py` 末尾追加：

```python
class SandboxViolationError(PureAgentLoopError):
    """沙箱权限违规异常

    当文件操作的目标路径不在沙箱允许范围内时抛出。
    """

    def __init__(self, path: str, operation: str, allowed_paths: list[str]):
        self.path = path
        self.operation = operation
        self.allowed_paths = allowed_paths
        super().__init__(
            f"沙箱权限不足: 无法对 '{path}' 执行 {operation} 操作。"
            f"允许的路径: {allowed_paths}"
        )
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_errors.py::TestSandboxViolationError -v`
Expected: PASS

**Step 5: 编写 Sandbox 和 SandboxGuard 的失败测试**

创建 `tests/test_sandbox.py`：

```python
"""沙箱系统测试"""

import pytest
from pathlib import Path
from pure_agent_loop.sandbox import Sandbox, SandboxGuard
from pure_agent_loop.errors import SandboxViolationError


class TestSandbox:
    """Sandbox 配置类测试"""

    def test_default_empty(self):
        """默认无任何路径"""
        sb = Sandbox()
        assert sb.read_paths == []
        assert sb.write_paths == []

    def test_paths_resolved_to_absolute(self, tmp_path):
        """路径应被 resolve 为绝对路径"""
        sb = Sandbox(read_paths=["./relative"], write_paths=[str(tmp_path)])
        assert all(p.is_absolute() for p in sb.read_paths)
        assert all(p.is_absolute() for p in sb.write_paths)

    def test_string_paths_converted_to_path(self, tmp_path):
        """字符串路径应转为 Path 对象"""
        sb = Sandbox(read_paths=[str(tmp_path)])
        assert isinstance(sb.read_paths[0], Path)


class TestSandboxGuard:
    """SandboxGuard 验证器测试"""

    def test_read_allowed_in_read_paths(self, tmp_path):
        """read_paths 内的文件应允许读取"""
        guard = SandboxGuard(Sandbox(read_paths=[str(tmp_path)]))
        guard.check_read(tmp_path / "file.txt")  # 不应抛异常

    def test_read_allowed_in_write_paths(self, tmp_path):
        """write_paths 隐含读权限"""
        guard = SandboxGuard(Sandbox(write_paths=[str(tmp_path)]))
        guard.check_read(tmp_path / "file.txt")  # 不应抛异常

    def test_read_denied_outside_sandbox(self, tmp_path):
        """沙箱外路径应拒绝读取"""
        guard = SandboxGuard(Sandbox(read_paths=[str(tmp_path / "allowed")]))
        with pytest.raises(SandboxViolationError):
            guard.check_read("/etc/passwd")

    def test_write_allowed_in_write_paths(self, tmp_path):
        """write_paths 内应允许写入"""
        guard = SandboxGuard(Sandbox(write_paths=[str(tmp_path)]))
        guard.check_write(tmp_path / "file.txt")  # 不应抛异常

    def test_write_denied_in_read_paths(self, tmp_path):
        """read_paths 应拒绝写入"""
        guard = SandboxGuard(Sandbox(read_paths=[str(tmp_path)]))
        with pytest.raises(SandboxViolationError):
            guard.check_write(tmp_path / "file.txt")

    def test_write_denied_outside_sandbox(self, tmp_path):
        """沙箱外路径应拒绝写入"""
        guard = SandboxGuard(Sandbox(write_paths=[str(tmp_path / "allowed")]))
        with pytest.raises(SandboxViolationError):
            guard.check_write("/tmp/evil.txt")

    def test_is_readable_returns_bool(self, tmp_path):
        """is_readable 应返回布尔值不抛异常"""
        guard = SandboxGuard(Sandbox(read_paths=[str(tmp_path)]))
        assert guard.is_readable(tmp_path / "file.txt") is True
        assert guard.is_readable("/etc/passwd") is False

    def test_symlink_escape_prevented(self, tmp_path):
        """符号链接逃逸应被阻止（resolve 后检查）"""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")
        link = allowed / "link.txt"
        link.symlink_to(secret)

        guard = SandboxGuard(Sandbox(read_paths=[str(allowed)]))
        with pytest.raises(SandboxViolationError):
            guard.check_read(link)

    def test_multiple_read_paths(self, tmp_path):
        """多个 read_paths 都应可读"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        guard = SandboxGuard(Sandbox(read_paths=[str(dir_a), str(dir_b)]))
        guard.check_read(dir_a / "file.txt")
        guard.check_read(dir_b / "file.txt")

    def test_empty_sandbox_denies_all(self):
        """空沙箱应拒绝所有操作"""
        guard = SandboxGuard(Sandbox())
        with pytest.raises(SandboxViolationError):
            guard.check_read("/any/file.txt")
        with pytest.raises(SandboxViolationError):
            guard.check_write("/any/file.txt")
```

**Step 6: 运行测试验证失败**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'pure_agent_loop.sandbox')

**Step 7: 实现 sandbox.py**

创建 `src/pure_agent_loop/sandbox.py`：

```python
"""沙箱系统

提供文件访问的路径权限控制。
"""

from dataclasses import dataclass, field
from pathlib import Path

from .errors import SandboxViolationError


@dataclass
class Sandbox:
    """沙箱配置

    定义文件工具可访问的路径范围。

    Attributes:
        read_paths: 仅允许读取的路径列表
        write_paths: 允许读写的路径列表（隐含读权限）
    """

    read_paths: list[str | Path] = field(default_factory=list)
    write_paths: list[str | Path] = field(default_factory=list)

    def __post_init__(self):
        """将所有路径 resolve 为绝对 Path 对象"""
        self.read_paths = [Path(p).resolve() for p in self.read_paths]
        self.write_paths = [Path(p).resolve() for p in self.write_paths]


class SandboxGuard:
    """沙箱路径验证器

    检查文件操作的目标路径是否在沙箱允许范围内。

    Args:
        sandbox: 沙箱配置实例
    """

    def __init__(self, sandbox: Sandbox):
        self._read_paths = sandbox.read_paths
        self._write_paths = sandbox.write_paths

    def _all_readable_paths(self) -> list[Path]:
        """所有可读路径（read_paths + write_paths）"""
        return self._read_paths + self._write_paths

    def _resolve(self, path: str | Path) -> Path:
        """解析路径为绝对路径"""
        return Path(path).resolve()

    def _is_under(self, target: Path, allowed: list[Path]) -> bool:
        """检查 target 是否在 allowed 中任一路径下"""
        for base in allowed:
            try:
                target.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    def check_read(self, path: str | Path) -> None:
        """检查读权限，失败抛出 SandboxViolationError"""
        resolved = self._resolve(path)
        if not self._is_under(resolved, self._all_readable_paths()):
            raise SandboxViolationError(
                str(path), "read",
                [str(p) for p in self._all_readable_paths()],
            )

    def check_write(self, path: str | Path) -> None:
        """检查写权限，失败抛出 SandboxViolationError"""
        resolved = self._resolve(path)
        if not self._is_under(resolved, self._write_paths):
            raise SandboxViolationError(
                str(path), "write",
                [str(p) for p in self._write_paths],
            )

    def is_readable(self, path: str | Path) -> bool:
        """检查路径是否可读（不抛异常）"""
        resolved = self._resolve(path)
        return self._is_under(resolved, self._all_readable_paths())
```

**Step 8: 运行测试验证通过**

Run: `pytest tests/test_sandbox.py -v`
Expected: PASS

**Step 9: 提交**

```bash
git add src/pure_agent_loop/sandbox.py src/pure_agent_loop/errors.py tests/test_sandbox.py tests/test_errors.py
git commit -m "feat: 添加 Sandbox 配置类和 SandboxGuard 路径验证器"
```

---

## Task 2: 实现 file_read 工具

**Files:**
- Create: `src/pure_agent_loop/file_tools.py`
- Create: `tests/test_file_tools.py`

**Step 1: 编写 file_read 失败测试**

创建 `tests/test_file_tools.py`：

```python
"""文件工具测试"""

import pytest
from pathlib import Path
from pure_agent_loop.sandbox import Sandbox, SandboxGuard
from pure_agent_loop.file_tools import create_file_tools
from pure_agent_loop.tool import Tool


@pytest.fixture
def sandbox_dirs(tmp_path):
    """创建沙箱测试目录结构"""
    read_dir = tmp_path / "read_only"
    write_dir = tmp_path / "writable"
    outside_dir = tmp_path / "outside"
    read_dir.mkdir()
    write_dir.mkdir()
    outside_dir.mkdir()
    return read_dir, write_dir, outside_dir


@pytest.fixture
def guard(sandbox_dirs):
    """创建 SandboxGuard"""
    read_dir, write_dir, _ = sandbox_dirs
    return SandboxGuard(Sandbox(
        read_paths=[str(read_dir)],
        write_paths=[str(write_dir)],
    ))


@pytest.fixture
def tools(guard):
    """创建文件工具列表"""
    return {t.name: t for t in create_file_tools(guard)}


class TestCreateFileTools:
    """create_file_tools 工厂函数测试"""

    def test_returns_five_tools(self, guard):
        """应返回 5 个工具"""
        tools = create_file_tools(guard)
        assert len(tools) == 5
        assert all(isinstance(t, Tool) for t in tools)

    def test_tool_names(self, guard):
        """工具名称应正确"""
        names = {t.name for t in create_file_tools(guard)}
        assert names == {"file_read", "file_search", "file_grep", "file_edit", "file_write"}


class TestFileRead:
    """file_read 工具测试"""

    async def test_read_file_in_read_paths(self, tools, sandbox_dirs):
        """应能读取 read_paths 内的文件"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await tools["file_read"].execute({"file_path": str(f)})
        assert "line1" in result
        assert "line2" in result

    async def test_read_file_in_write_paths(self, tools, sandbox_dirs):
        """应能读取 write_paths 内的文件（隐含读权限）"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "data.txt"
        f.write_text("content")
        result = await tools["file_read"].execute({"file_path": str(f)})
        assert "content" in result

    async def test_read_denied_outside_sandbox(self, tools, sandbox_dirs):
        """沙箱外文件应返回权限错误"""
        _, _, outside_dir = sandbox_dirs
        f = outside_dir / "secret.txt"
        f.write_text("secret")
        result = await tools["file_read"].execute({"file_path": str(f)})
        assert "权限" in result or "沙箱" in result

    async def test_read_nonexistent_file(self, tools, sandbox_dirs):
        """不存在的文件应返回错误"""
        read_dir, _, _ = sandbox_dirs
        result = await tools["file_read"].execute({"file_path": str(read_dir / "nope.txt")})
        assert "不存在" in result or "找不到" in result

    async def test_read_with_line_numbers(self, tools, sandbox_dirs):
        """返回内容应带行号"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "numbered.txt"
        f.write_text("aaa\nbbb\nccc\n")
        result = await tools["file_read"].execute({"file_path": str(f)})
        assert "1\t" in result or "1 " in result

    async def test_read_with_offset_and_limit(self, tools, sandbox_dirs):
        """应支持 offset 和 limit 参数"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "long.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 101)))
        result = await tools["file_read"].execute({
            "file_path": str(f), "offset": 10, "limit": 5,
        })
        assert "line10" in result
        assert "line14" in result
        assert "line15" not in result

    async def test_read_directory_lists_entries(self, tools, sandbox_dirs):
        """读取目录应返回目录列表"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "a.txt").write_text("a")
        (read_dir / "b.txt").write_text("b")
        result = await tools["file_read"].execute({"file_path": str(read_dir)})
        assert "a.txt" in result
        assert "b.txt" in result

    async def test_read_binary_file_rejected(self, tools, sandbox_dirs):
        """二进制文件应拒绝读取"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        result = await tools["file_read"].execute({"file_path": str(f)})
        assert "二进制" in result

    async def test_read_truncates_long_lines(self, tools, sandbox_dirs):
        """超长行应被截断"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "long_line.txt"
        f.write_text("x" * 3000 + "\n")
        result = await tools["file_read"].execute({"file_path": str(f)})
        # 行内容不应超过 2000 字符 + 行号前缀
        lines = result.strip().split("\n")
        for line in lines:
            if line.startswith(("⚠", "📁")):
                continue
            assert len(line) <= 2100  # 2000 + 行号 + tab + 余量
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_tools.py::TestFileRead -v`
Expected: FAIL (ModuleNotFoundError: No module named 'pure_agent_loop.file_tools')

**Step 3: 实现 file_tools.py（file_read 部分）**

创建 `src/pure_agent_loop/file_tools.py`：

```python
"""文件工具

提供文件搜索、读取和编辑工具，配合沙箱权限系统使用。
"""

from pathlib import Path

from .sandbox import SandboxGuard
from .tool import Tool

# 常量
DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000

# 二进制文件扩展名黑名单
BINARY_EXTENSIONS = frozenset({
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".class", ".o", ".obj",
    ".woff", ".woff2", ".ttf", ".eot",
    ".sqlite", ".db",
})


def _is_binary(path: Path) -> bool:
    """检测文件是否为二进制文件"""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        chunk = path.read_bytes()[:4096]
        if b"\x00" in chunk:
            return True
        # 超过 30% 不可打印字符视为二进制
        non_printable = sum(
            1 for b in chunk
            if b < 32 and b not in (9, 10, 13)  # tab, LF, CR
        )
        if len(chunk) > 0 and non_printable / len(chunk) > 0.3:
            return True
    except OSError:
        return False
    return False


def _read_file(path: Path, offset: int | None, limit: int | None) -> str:
    """读取文件内容，返回带行号的文本"""
    limit = limit or DEFAULT_READ_LIMIT
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    start = (offset - 1) if offset and offset > 0 else 0
    end = start + limit
    selected = lines[start:end]

    result_lines = []
    for i, line in enumerate(selected, start=start + 1):
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "... (截断)"
        result_lines.append(f"{i}\t{line}")

    output = "\n".join(result_lines)
    if end < len(lines):
        output += f"\n\n(共 {len(lines)} 行，已显示 {start + 1}-{min(end, len(lines))} 行)"
    return output


def _list_directory(path: Path) -> str:
    """列出目录内容"""
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    lines = [f"📁 目录: {path}"]
    for entry in entries[:100]:
        prefix = "📂" if entry.is_dir() else "📄"
        lines.append(f"  {prefix} {entry.name}")
    if len(list(path.iterdir())) > 100:
        lines.append(f"  ... (共 {len(list(path.iterdir()))} 项，仅显示前 100 项)")
    return "\n".join(lines)


def create_file_tools(guard: SandboxGuard) -> list[Tool]:
    """创建文件工具列表

    通过闭包捕获 SandboxGuard 实例，实现路径权限控制。

    Args:
        guard: 沙箱路径验证器

    Returns:
        包含 5 个文件工具的列表
    """

    def file_read(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        """读取文件内容

        Args:
            file_path: 文件的绝对路径
            offset: 起始行号（从 1 开始），不指定则从头读取
            limit: 最多读取的行数，默认 2000
        """
        path = Path(file_path).resolve()

        # 权限检查
        guard.check_read(path)

        if not path.exists():
            return f"⚠️ 文件不存在: '{file_path}'"

        # 目录处理
        if path.is_dir():
            return _list_directory(path)

        # 二进制检测
        if _is_binary(path):
            return f"⚠️ 无法读取二进制文件: '{file_path}'"

        return _read_file(path, offset, limit)

    # 占位：后续 Task 实现
    def file_search(pattern: str, path: str | None = None) -> str:
        """按文件名模式搜索文件

        Args:
            pattern: glob 模式，如 '**/*.py'
            path: 搜索起始目录
        """
        return "⚠️ file_search 尚未实现"

    def file_grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
        """按正则表达式搜索文件内容

        Args:
            pattern: 正则表达式
            path: 搜索目录
            include: 文件名过滤模式，如 '*.py'
        """
        return "⚠️ file_grep 尚未实现"

    def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """精确替换文件中的文本

        Args:
            file_path: 文件路径
            old_string: 要替换的原始文本（必须精确匹配）
            new_string: 替换后的文本
            replace_all: 是否替换所有匹配项，默认仅替换第一个
        """
        return "⚠️ file_edit 尚未实现"

    def file_write(file_path: str, content: str) -> str:
        """创建新文件或完全重写已有文件

        Args:
            file_path: 文件路径
            content: 完整的文件内容
        """
        return "⚠️ file_write 尚未实现"

    # 构建 Tool 对象列表
    from .tool import _build_tool
    return [
        _build_tool(file_read),
        _build_tool(file_search),
        _build_tool(file_grep),
        _build_tool(file_edit),
        _build_tool(file_write),
    ]
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_file_tools.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat: 添加 file_read 工具和文件工具框架"
```

---

## Task 3: 实现 file_search 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Modify: `tests/test_file_tools.py`

**Step 1: 在 test_file_tools.py 末尾追加 file_search 测试**

```python
class TestFileSearch:
    """file_search 工具测试"""

    async def test_search_finds_files(self, tools, sandbox_dirs):
        """应能找到匹配的文件"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "main.py").write_text("print('hello')")
        (read_dir / "utils.py").write_text("pass")
        (read_dir / "readme.md").write_text("# readme")
        result = await tools["file_search"].execute({
            "pattern": "*.py", "path": str(read_dir),
        })
        assert "main.py" in result
        assert "utils.py" in result
        assert "readme.md" not in result

    async def test_search_recursive(self, tools, sandbox_dirs):
        """应支持递归搜索"""
        read_dir, _, _ = sandbox_dirs
        sub = read_dir / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("pass")
        result = await tools["file_search"].execute({
            "pattern": "**/*.py", "path": str(read_dir),
        })
        assert "deep.py" in result

    async def test_search_denied_outside_sandbox(self, tools, sandbox_dirs):
        """沙箱外目录应拒绝搜索"""
        _, _, outside_dir = sandbox_dirs
        result = await tools["file_search"].execute({
            "pattern": "*.py", "path": str(outside_dir),
        })
        assert "权限" in result or "沙箱" in result

    async def test_search_skips_ignored_dirs(self, tools, sandbox_dirs):
        """应跳过 .git、__pycache__ 等目录"""
        read_dir, _, _ = sandbox_dirs
        git_dir = read_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")
        (read_dir / "main.py").write_text("pass")
        result = await tools["file_search"].execute({
            "pattern": "**/*", "path": str(read_dir),
        })
        assert "config" not in result or ".git" not in result

    async def test_search_max_results(self, tools, sandbox_dirs):
        """结果应限制在 100 个以内"""
        read_dir, _, _ = sandbox_dirs
        for i in range(120):
            (read_dir / f"file_{i}.txt").write_text(f"content {i}")
        result = await tools["file_search"].execute({
            "pattern": "*.txt", "path": str(read_dir),
        })
        assert "100" in result or "截断" in result or "truncated" in result

    async def test_search_default_path(self, tools, sandbox_dirs):
        """未指定 path 时应使用第一个可读路径"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "default.py").write_text("pass")
        result = await tools["file_search"].execute({"pattern": "*.py"})
        assert "default.py" in result

    async def test_search_no_matches(self, tools, sandbox_dirs):
        """无匹配时应返回提示"""
        read_dir, _, _ = sandbox_dirs
        result = await tools["file_search"].execute({
            "pattern": "*.xyz", "path": str(read_dir),
        })
        assert "未找到" in result or "0" in result
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_tools.py::TestFileSearch -v`
Expected: FAIL (file_search 返回 "尚未实现")

**Step 3: 在 file_tools.py 中实现 file_search**

替换 `file_search` 占位函数为：

```python
    # 忽略的目录名
    IGNORED_DIRS = frozenset({
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        ".idea", ".vscode", ".eggs", "*.egg-info",
    })

    def file_search(pattern: str, path: str | None = None) -> str:
        """按文件名模式搜索文件

        Args:
            pattern: glob 模式，如 '**/*.py'
            path: 搜索起始目录，默认为第一个可读路径
        """
        if path:
            search_path = Path(path).resolve()
            guard.check_read(search_path)
        else:
            readable = guard._all_readable_paths()
            if not readable:
                return "⚠️ 沙箱中没有可读路径"
            search_path = readable[0]

        if not search_path.exists() or not search_path.is_dir():
            return f"⚠️ 目录不存在: '{path}'"

        matches = []
        for p in search_path.glob(pattern):
            # 跳过忽略目录
            if any(part in IGNORED_DIRS for part in p.parts):
                continue
            if not p.is_file():
                continue
            if not guard.is_readable(p):
                continue
            matches.append(p)
            if len(matches) >= 100:
                break

        if not matches:
            return f"未找到匹配 '{pattern}' 的文件"

        # 按修改时间降序排列
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        lines = [f"找到 {len(matches)} 个文件:"]
        for m in matches:
            try:
                rel = m.relative_to(search_path)
            except ValueError:
                rel = m
            lines.append(f"  {rel}")

        if len(matches) >= 100:
            lines.append("  ... (结果已截断，仅显示前 100 个)")

        return "\n".join(lines)
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_file_tools.py::TestFileSearch -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat: 实现 file_search (glob) 文件搜索工具"
```

---

## Task 4: 实现 file_grep 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Modify: `tests/test_file_tools.py`

**Step 1: 在 test_file_tools.py 末尾追加 file_grep 测试**

```python
class TestFileGrep:
    """file_grep 工具测试"""

    async def test_grep_finds_matches(self, tools, sandbox_dirs):
        """应能找到匹配内容"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "main.py").write_text("def hello():\n    print('world')\n")
        result = await tools["file_grep"].execute({
            "pattern": "def hello", "path": str(read_dir),
        })
        assert "main.py" in result
        assert "def hello" in result

    async def test_grep_with_line_numbers(self, tools, sandbox_dirs):
        """结果应包含行号"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "test.py").write_text("aaa\nbbb\nccc\n")
        result = await tools["file_grep"].execute({
            "pattern": "bbb", "path": str(read_dir),
        })
        assert "2" in result  # 第 2 行

    async def test_grep_with_include_filter(self, tools, sandbox_dirs):
        """include 参数应过滤文件类型"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "code.py").write_text("hello world")
        (read_dir / "note.md").write_text("hello world")
        result = await tools["file_grep"].execute({
            "pattern": "hello", "path": str(read_dir), "include": "*.py",
        })
        assert "code.py" in result
        assert "note.md" not in result

    async def test_grep_regex_support(self, tools, sandbox_dirs):
        """应支持正则表达式"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "data.txt").write_text("foo123bar\nfoo456bar\nhello\n")
        result = await tools["file_grep"].execute({
            "pattern": r"foo\d+bar", "path": str(read_dir),
        })
        assert "foo123bar" in result
        assert "foo456bar" in result
        assert "hello" not in result

    async def test_grep_denied_outside_sandbox(self, tools, sandbox_dirs):
        """沙箱外目录应拒绝搜索"""
        _, _, outside_dir = sandbox_dirs
        result = await tools["file_grep"].execute({
            "pattern": "secret", "path": str(outside_dir),
        })
        assert "权限" in result or "沙箱" in result

    async def test_grep_no_matches(self, tools, sandbox_dirs):
        """无匹配时应返回提示"""
        read_dir, _, _ = sandbox_dirs
        (read_dir / "empty.txt").write_text("nothing here")
        result = await tools["file_grep"].execute({
            "pattern": "xyz_not_found", "path": str(read_dir),
        })
        assert "未找到" in result or "0" in result

    async def test_grep_max_results(self, tools, sandbox_dirs):
        """结果应限制在 100 个以内"""
        read_dir, _, _ = sandbox_dirs
        content = "\n".join(f"match_line_{i}" for i in range(150))
        (read_dir / "big.txt").write_text(content)
        result = await tools["file_grep"].execute({
            "pattern": "match_line_", "path": str(read_dir),
        })
        assert "100" in result or "截断" in result
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_tools.py::TestFileGrep -v`
Expected: FAIL

**Step 3: 在 file_tools.py 中实现 file_grep**

替换 `file_grep` 占位函数为：

```python
    def file_grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
        """按正则表达式搜索文件内容

        Args:
            pattern: 正则表达式
            path: 搜索目录，默认为第一个可读路径
            include: 文件名过滤模式，如 '*.py'
        """
        import re
        import fnmatch

        if path:
            search_path = Path(path).resolve()
            guard.check_read(search_path)
        else:
            readable = guard._all_readable_paths()
            if not readable:
                return "⚠️ 沙箱中没有可读路径"
            search_path = readable[0]

        if not search_path.exists() or not search_path.is_dir():
            return f"⚠️ 目录不存在: '{path}'"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"⚠️ 无效的正则表达式: {e}"

        matches = []
        max_matches = 100

        for file_path in search_path.rglob("*"):
            if len(matches) >= max_matches:
                break
            if not file_path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in file_path.parts):
                continue
            if not guard.is_readable(file_path):
                continue
            if include and not fnmatch.fnmatch(file_path.name, include):
                continue
            if _is_binary(file_path):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    try:
                        rel = file_path.relative_to(search_path)
                    except ValueError:
                        rel = file_path
                    display_line = line[:200] if len(line) > 200 else line
                    matches.append(f"{rel}:{line_num}: {display_line}")
                    if len(matches) >= max_matches:
                        break

        if not matches:
            return f"未找到匹配 '{pattern}' 的内容"

        header = f"找到 {len(matches)} 个匹配:"
        if len(matches) >= max_matches:
            header += " (结果已截断，仅显示前 100 个)"
        return header + "\n" + "\n".join(matches)
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_file_tools.py::TestFileGrep -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat: 实现 file_grep 正则内容搜索工具"
```

---

## Task 5: 实现 file_edit 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Modify: `tests/test_file_tools.py`

**Step 1: 在 test_file_tools.py 末尾追加 file_edit 测试**

```python
class TestFileEdit:
    """file_edit 工具测试"""

    async def test_edit_replaces_text(self, tools, sandbox_dirs):
        """应能精确替换文本"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "code.py"
        f.write_text("def hello():\n    pass\n")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "    pass",
            "new_string": "    print('hello')",
        })
        assert f.read_text() == "def hello():\n    print('hello')\n"
        assert "diff" in result.lower() or "替换" in result or "---" in result

    async def test_edit_denied_in_read_paths(self, tools, sandbox_dirs):
        """read_paths 内的文件应拒绝编辑"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "readonly.py"
        f.write_text("original")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "original",
            "new_string": "modified",
        })
        assert "权限" in result or "沙箱" in result
        assert f.read_text() == "original"  # 文件未被修改

    async def test_edit_denied_outside_sandbox(self, tools, sandbox_dirs):
        """沙箱外文件应拒绝编辑"""
        _, _, outside_dir = sandbox_dirs
        f = outside_dir / "secret.py"
        f.write_text("secret")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "secret",
            "new_string": "hacked",
        })
        assert "权限" in result or "沙箱" in result

    async def test_edit_no_match(self, tools, sandbox_dirs):
        """未找到匹配时应返回错误"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "code.py"
        f.write_text("hello world")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "not_exist",
            "new_string": "replaced",
        })
        assert "未找到" in result
        assert f.read_text() == "hello world"  # 文件未被修改

    async def test_edit_multiple_matches_error(self, tools, sandbox_dirs):
        """多处匹配且 replace_all=False 时应报错"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "dup.py"
        f.write_text("aaa\nbbb\naaa\n")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "aaa",
            "new_string": "ccc",
        })
        assert "多处" in result or "匹配" in result
        assert f.read_text() == "aaa\nbbb\naaa\n"  # 文件未被修改

    async def test_edit_replace_all(self, tools, sandbox_dirs):
        """replace_all=True 应替换所有匹配"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "dup.py"
        f.write_text("aaa\nbbb\naaa\n")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "aaa",
            "new_string": "ccc",
            "replace_all": True,
        })
        assert f.read_text() == "ccc\nbbb\nccc\n"

    async def test_edit_nonexistent_file(self, tools, sandbox_dirs):
        """不存在的文件应返回错误"""
        _, write_dir, _ = sandbox_dirs
        result = await tools["file_edit"].execute({
            "file_path": str(write_dir / "nope.py"),
            "old_string": "a",
            "new_string": "b",
        })
        assert "不存在" in result

    async def test_edit_returns_diff(self, tools, sandbox_dirs):
        """应返回 diff 格式的变更摘要"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "diff_test.py"
        f.write_text("old_line\n")
        result = await tools["file_edit"].execute({
            "file_path": str(f),
            "old_string": "old_line",
            "new_string": "new_line",
        })
        assert "old_line" in result
        assert "new_line" in result
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_tools.py::TestFileEdit -v`
Expected: FAIL

**Step 3: 在 file_tools.py 中实现 file_edit**

替换 `file_edit` 占位函数为：

```python
    def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """精确替换文件中的文本

        Args:
            file_path: 文件路径
            old_string: 要替换的原始文本（必须精确匹配）
            new_string: 替换后的文本
            replace_all: 是否替换所有匹配项，默认仅替换第一个
        """
        import difflib

        path = Path(file_path).resolve()
        guard.check_write(path)

        if not path.exists():
            return f"⚠️ 文件不存在: '{file_path}'"

        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)

        if count == 0:
            return f"⚠️ 未找到匹配内容，请检查 old_string 是否与文件内容完全一致"

        if count > 1 and not replace_all:
            return (
                f"⚠️ 找到 {count} 处匹配，请提供更多上下文使匹配唯一，"
                f"或设置 replace_all=True"
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        # 生成 diff
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
        diff_text = "".join(diff)

        # 写入文件
        path.write_text(new_content, encoding="utf-8")

        replaced = count if replace_all else 1
        return f"✅ 已替换 {replaced} 处匹配\n\n{diff_text}"
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_file_tools.py::TestFileEdit -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat: 实现 file_edit 精确字符串替换工具"
```

---

## Task 6: 实现 file_write 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Modify: `tests/test_file_tools.py`

**Step 1: 在 test_file_tools.py 末尾追加 file_write 测试**

```python
class TestFileWrite:
    """file_write 工具测试"""

    async def test_write_creates_new_file(self, tools, sandbox_dirs):
        """应能创建新文件"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "new.py"
        result = await tools["file_write"].execute({
            "file_path": str(f),
            "content": "print('hello')\n",
        })
        assert f.exists()
        assert f.read_text() == "print('hello')\n"
        assert "新建" in result or "创建" in result

    async def test_write_overwrites_existing(self, tools, sandbox_dirs):
        """应能覆盖已有文件"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "exist.py"
        f.write_text("old content")
        result = await tools["file_write"].execute({
            "file_path": str(f),
            "content": "new content",
        })
        assert f.read_text() == "new content"
        assert "diff" in result.lower() or "---" in result

    async def test_write_creates_parent_dirs(self, tools, sandbox_dirs):
        """应自动创建不存在的父目录"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "deep" / "nested" / "file.py"
        await tools["file_write"].execute({
            "file_path": str(f),
            "content": "pass\n",
        })
        assert f.exists()
        assert f.read_text() == "pass\n"

    async def test_write_denied_in_read_paths(self, tools, sandbox_dirs):
        """read_paths 内应拒绝写入"""
        read_dir, _, _ = sandbox_dirs
        f = read_dir / "readonly.py"
        result = await tools["file_write"].execute({
            "file_path": str(f),
            "content": "hacked",
        })
        assert "权限" in result or "沙箱" in result
        assert not f.exists()

    async def test_write_denied_outside_sandbox(self, tools, sandbox_dirs):
        """沙箱外应拒绝写入"""
        _, _, outside_dir = sandbox_dirs
        result = await tools["file_write"].execute({
            "file_path": str(outside_dir / "evil.py"),
            "content": "evil",
        })
        assert "权限" in result or "沙箱" in result

    async def test_write_returns_diff_for_existing(self, tools, sandbox_dirs):
        """覆盖已有文件时应返回 diff"""
        _, write_dir, _ = sandbox_dirs
        f = write_dir / "diff.py"
        f.write_text("line1\nline2\n")
        result = await tools["file_write"].execute({
            "file_path": str(f),
            "content": "line1\nline3\n",
        })
        assert "line2" in result
        assert "line3" in result
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_tools.py::TestFileWrite -v`
Expected: FAIL

**Step 3: 在 file_tools.py 中实现 file_write**

替换 `file_write` 占位函数为：

```python
    def file_write(file_path: str, content: str) -> str:
        """创建新文件或完全重写已有文件

        Args:
            file_path: 文件路径
            content: 完整的文件内容
        """
        import difflib

        path = Path(file_path).resolve()
        guard.check_write(path)

        is_new = not path.exists()
        old_content = ""
        if not is_new:
            old_content = path.read_text(encoding="utf-8", errors="replace")

        # 创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        path.write_text(content, encoding="utf-8")

        if is_new:
            line_count = len(content.splitlines())
            return f"✅ 新建文件: {file_path} ({line_count} 行)"

        # 生成 diff
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
        diff_text = "".join(diff)
        return f"✅ 已重写文件: {file_path}\n\n{diff_text}"
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_file_tools.py::TestFileWrite -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat: 实现 file_write 文件创建/重写工具"
```

---

## Task 7: Agent 集成 + 公开 API 导出

**Files:**
- Modify: `src/pure_agent_loop/agent.py:20,79-119`
- Modify: `src/pure_agent_loop/__init__.py:22,29-67`
- Modify: `tests/test_agent.py`

**Step 1: 在 test_agent.py 末尾追加集成测试**

```python
from pure_agent_loop.sandbox import Sandbox


class TestAgentSandboxIntegration:
    """Agent 沙箱集成测试"""

    def test_sandbox_none_no_file_tools(self):
        """sandbox=None 时不应注册文件工具"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        assert agent._tool_registry.get("file_read") is None
        assert agent._tool_registry.get("file_write") is None

    def test_sandbox_registers_file_tools(self, tmp_path):
        """sandbox 存在时应自动注册 5 个文件工具"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(
            llm=mock_llm,
            sandbox=Sandbox(
                read_paths=[str(tmp_path / "read")],
                write_paths=[str(tmp_path / "write")],
            ),
        )
        assert agent._tool_registry.get("file_read") is not None
        assert agent._tool_registry.get("file_search") is not None
        assert agent._tool_registry.get("file_grep") is not None
        assert agent._tool_registry.get("file_edit") is not None
        assert agent._tool_registry.get("file_write") is not None

    def test_sandbox_coexists_with_user_tools(self, tmp_path):
        """文件工具应与用户自定义工具共存"""

        @tool
        def my_tool(query: str) -> str:
            """自定义工具"""
            return "result"

        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(
            llm=mock_llm,
            tools=[my_tool],
            sandbox=Sandbox(write_paths=[str(tmp_path)]),
        )
        assert agent._tool_registry.get("my_tool") is not None
        assert agent._tool_registry.get("file_read") is not None
        assert agent._tool_registry.get("todo_write") is not None

    async def test_file_read_via_agent(self, tmp_path):
        """通过 Agent 执行 file_read 应正常工作"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello from agent")

        mock_llm = MockLLM([
            _tool_call_response("file_read", {"file_path": str(test_file)}),
            _text_response("文件内容是 hello from agent"),
        ])
        agent = Agent(
            llm=mock_llm,
            sandbox=Sandbox(read_paths=[str(tmp_path)]),
        )
        result = await agent.arun("读取文件")
        assert result.content == "文件内容是 hello from agent"
        action_events = [e for e in result.events if e.type == EventType.ACTION]
        assert any(e.data["tool"] == "file_read" for e in action_events)
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_agent.py::TestAgentSandboxIntegration -v`
Expected: FAIL (Agent 没有 sandbox 参数)

**Step 3: 修改 agent.py 添加 sandbox 参数**

在 `src/pure_agent_loop/agent.py` 中：

1. 添加导入（第 20 行后）：
```python
from .sandbox import Sandbox, SandboxGuard
from .file_tools import create_file_tools
```

2. 在 `__init__` 参数列表中添加 `sandbox: Sandbox | None = None`（在 `skills_dir` 之前）

3. 在工具注册逻辑中（第 116-119 行之间），添加沙箱工具注册：
```python
        # 注册文件工具（如果配置了沙箱）
        if sandbox:
            file_guard = SandboxGuard(sandbox)
            file_tools = create_file_tools(file_guard)
            self._tool_registry.register_many(file_tools)
```

**Step 4: 修改 __init__.py 添加导出**

在 `src/pure_agent_loop/__init__.py` 中：

1. 添加导入：
```python
from .sandbox import Sandbox, SandboxGuard, SandboxViolationError
```

2. 在 `__all__` 中添加：
```python
    # 沙箱
    "Sandbox",
    "SandboxGuard",
    "SandboxViolationError",
```

3. 在 errors 导入中添加 `SandboxViolationError`：
```python
from .errors import (
    PureAgentLoopError,
    ToolExecutionError,
    LLMError,
    LimitExceededError,
    SandboxViolationError,
)
```

**Step 5: 运行全部测试验证通过**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 6: 提交**

```bash
git add src/pure_agent_loop/agent.py src/pure_agent_loop/__init__.py tests/test_agent.py
git commit -m "feat: Agent 集成沙箱系统，sandbox 参数自动注册文件工具"
```

---

## Task 8: 全量测试 + 最终验证

**Step 1: 运行全部测试（含覆盖率）**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing -v`
Expected: ALL PASS，新增模块覆盖率 > 80%

**Step 2: 验证导入正常**

Run: `python -c "from pure_agent_loop import Sandbox, SandboxGuard, SandboxViolationError; print('导入成功')"`
Expected: 输出 "导入成功"

**Step 3: 最终提交（如有修复）**

```bash
git add -A
git commit -m "test: 补充文件工具和沙箱系统测试覆盖"
```
