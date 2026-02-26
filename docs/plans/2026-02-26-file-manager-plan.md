# 文件管理器实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 新增 file_tree 和 file_manage 两个文件工具，并增强 Sandbox 支持 cwd 工作目录。

**Architecture:** 在现有 Sandbox 中增加 cwd 字段（自动加入 write_paths），在 file_tools.py 中新增两个工具函数通过闭包捕获 SandboxGuard 和 cwd，Agent 在初始化时将 cwd 注入系统提示词。所有路径操作统一通过 SandboxGuard 进行权限校验。

**Tech Stack:** Python 3.10+, pathlib, shutil, pytest, pytest-asyncio

---

### Task 1: Sandbox 增加 cwd 字段

**Files:**
- Modify: `src/pure_agent_loop/sandbox.py:12-29`
- Test: `tests/test_sandbox.py`

**Step 1: 编写失败测试**

在 `tests/test_sandbox.py` 的 `TestSandbox` 类末尾添加：

```python
def test_cwd_auto_added_to_write_paths(self, tmp_path):
    """cwd 应自动加入 write_paths"""
    sb = Sandbox(cwd=str(tmp_path))
    assert any(p == tmp_path.resolve() for p in sb.write_paths)

def test_cwd_resolved_to_absolute(self, tmp_path):
    """cwd 应被 resolve 为绝对路径"""
    sb = Sandbox(cwd="./relative")
    assert sb.cwd.is_absolute()

def test_cwd_minimal_usage(self, tmp_path):
    """仅指定 cwd 即可获得读写权限"""
    sb = Sandbox(cwd=str(tmp_path))
    guard = SandboxGuard(sb)
    guard.check_read(tmp_path / "file.txt")
    guard.check_write(tmp_path / "file.txt")

def test_cwd_with_extra_paths(self, tmp_path):
    """cwd + 额外路径应共存"""
    extra = tmp_path / "extra"
    extra.mkdir()
    sb = Sandbox(
        cwd=str(tmp_path / "work"),
        read_paths=[str(extra)],
    )
    assert sb.cwd == (tmp_path / "work").resolve()
    assert any(p == extra.resolve() for p in sb.read_paths)

def test_default_sandbox_no_cwd(self):
    """默认 Sandbox 无 cwd 时 cwd 应为 None"""
    sb = Sandbox()
    assert sb.cwd is None
```

**Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && pytest tests/test_sandbox.py -v`
Expected: 新增的 5 个测试全部 FAIL

**Step 3: 实现 Sandbox cwd 字段**

修改 `src/pure_agent_loop/sandbox.py` 中 `Sandbox` 类：

```python
@dataclass
class Sandbox:
    """沙箱配置

    定义文件工具可访问的路径范围。

    Attributes:
        cwd: 工作目录（自动加入 write_paths），默认为 None
        read_paths: 仅允许读取的路径列表
        write_paths: 允许读写的路径列表（隐含读权限）
    """

    cwd: str | Path | None = None
    read_paths: list[str | Path] = field(default_factory=list)
    write_paths: list[str | Path] = field(default_factory=list)

    def __post_init__(self):
        """将所有路径 resolve 为绝对 Path 对象，cwd 自动加入 write_paths"""
        self.read_paths = [Path(p).resolve() for p in self.read_paths]
        self.write_paths = [Path(p).resolve() for p in self.write_paths]
        if self.cwd is not None:
            self.cwd = Path(self.cwd).resolve()
            # cwd 自动拥有读写权限
            if self.cwd not in self.write_paths:
                self.write_paths.append(self.cwd)
```

**Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && pytest tests/test_sandbox.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/sandbox.py tests/test_sandbox.py
git commit -m "feat(sandbox): 添加 cwd 工作目录字段，自动加入 write_paths"
```

---

### Task 2: 实现 file_tree 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Test: `tests/test_file_tools.py`

**Step 1: 编写失败测试**

在 `tests/test_file_tools.py` 末尾新增 `TestFileTree` 类。注意 fixture 需要调整以支持 cwd，先添加新 fixture：

```python
@pytest.fixture
def cwd_guard(tmp_path):
    """创建带 cwd 的 SandboxGuard"""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    return SandboxGuard(Sandbox(cwd=str(work_dir))), work_dir


@pytest.fixture
def cwd_tools(cwd_guard):
    """创建带 cwd 的文件工具"""
    guard, cwd = cwd_guard
    return {t.name: t for t in create_file_tools(guard, cwd=cwd)}, cwd


class TestFileTree:
    """file_tree 工具测试"""

    async def test_tree_empty_directory(self, cwd_tools):
        """空目录应返回空树"""
        tools, cwd = cwd_tools
        result = await tools["file_tree"].execute({"path": str(cwd)})
        assert "0 个文件" in result or "0 个目录" in result

    async def test_tree_with_files_and_dirs(self, cwd_tools):
        """应正确展示文件和目录结构"""
        tools, cwd = cwd_tools
        (cwd / "file1.txt").write_text("hello")
        (cwd / "subdir").mkdir()
        (cwd / "subdir" / "file2.txt").write_text("world")
        result = await tools["file_tree"].execute({"path": str(cwd)})
        assert "file1.txt" in result
        assert "subdir" in result
        assert "file2.txt" in result

    async def test_tree_max_depth(self, cwd_tools):
        """max_depth 应限制递归深度"""
        tools, cwd = cwd_tools
        deep = cwd / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")
        result = await tools["file_tree"].execute({
            "path": str(cwd), "max_depth": 1,
        })
        assert "a" in result
        assert "[...]" in result
        assert "deep.txt" not in result

    async def test_tree_ignores_special_dirs(self, cwd_tools):
        """应忽略 .git、__pycache__ 等目录"""
        tools, cwd = cwd_tools
        (cwd / ".git").mkdir()
        (cwd / ".git" / "config").write_text("cfg")
        (cwd / "src").mkdir()
        (cwd / "src" / "main.py").write_text("pass")
        result = await tools["file_tree"].execute({"path": str(cwd)})
        assert ".git" not in result
        assert "main.py" in result

    async def test_tree_default_path_uses_cwd(self, cwd_tools):
        """未指定 path 时应使用 cwd"""
        tools, cwd = cwd_tools
        (cwd / "default.txt").write_text("hello")
        result = await tools["file_tree"].execute({})
        assert "default.txt" in result

    async def test_tree_sandbox_denied(self, cwd_tools):
        """沙箱外路径应被拒绝"""
        tools, cwd = cwd_tools
        result = await tools["file_tree"].execute({"path": "/etc"})
        assert "权限" in result or "沙箱" in result

    async def test_tree_shows_summary(self, cwd_tools):
        """应显示目录和文件数量统计"""
        tools, cwd = cwd_tools
        (cwd / "a.txt").write_text("a")
        (cwd / "b.txt").write_text("b")
        (cwd / "sub").mkdir()
        result = await tools["file_tree"].execute({"path": str(cwd)})
        assert "1 个目录" in result
        assert "2 个文件" in result

    async def test_tree_uses_tree_format(self, cwd_tools):
        """应使用树形符号（├── └──）"""
        tools, cwd = cwd_tools
        (cwd / "a.txt").write_text("a")
        (cwd / "b.txt").write_text("b")
        result = await tools["file_tree"].execute({"path": str(cwd)})
        assert "├──" in result or "└──" in result
```

**Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py::TestFileTree -v`
Expected: 全部 FAIL（file_tree 工具不存在 / create_file_tools 签名不匹配）

**Step 3: 修改 create_file_tools 签名并实现 file_tree**

修改 `src/pure_agent_loop/file_tools.py`：

1. `create_file_tools` 签名改为 `create_file_tools(guard: SandboxGuard, cwd: Path | None = None) -> list[Tool]`
2. 在函数内部新增 `_build_tree` 辅助函数和 `file_tree` 工具函数
3. 将 `IGNORED_DIRS` 从 `file_search` 闭包提升为模块级常量（`file_tree` 也需要）

```python
# 模块级常量（从 file_search 内部提升）
IGNORED_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".idea", ".vscode", ".eggs", "*.egg-info",
})


def create_file_tools(guard: SandboxGuard, cwd: Path | None = None) -> list[Tool]:
    """创建文件工具列表

    Args:
        guard: 沙箱路径验证器
        cwd: 工作目录，用于解析相对路径和作为默认路径

    Returns:
        包含 7 个文件工具的列表
    """

    def _resolve_path(path_str: str | None, default: Path | None = None) -> Path:
        """解析路径：有 cwd 时相对路径基于 cwd 解析，否则直接 resolve"""
        if path_str is None:
            if default is not None:
                return default
            if cwd is not None:
                return cwd
            readable = guard._all_readable_paths()
            if readable:
                return readable[0]
            raise ValueError("没有可用的默认路径")
        p = Path(path_str)
        if not p.is_absolute() and cwd is not None:
            return (cwd / p).resolve()
        return p.resolve()

    # ... 现有工具函数中的路径解析改用 _resolve_path ...

    def _build_tree(
        dir_path: Path,
        prefix: str,
        max_depth: int,
        current_depth: int,
        stats: dict,
    ) -> list[str]:
        """递归构建树形结构的行列表"""
        if current_depth > max_depth:
            return [f"{prefix}[...]"]

        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        # 过滤忽略目录
        entries = [e for e in entries if e.name not in IGNORED_DIRS]

        lines = []
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "

            if entry.is_dir():
                stats["dirs"] += 1
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                lines.extend(
                    _build_tree(entry, prefix + extension, max_depth, current_depth + 1, stats)
                )
            else:
                stats["files"] += 1
                lines.append(f"{prefix}{connector}{entry.name}")

        return lines

    def file_tree(path: str | None = None, max_depth: int | None = None) -> str:
        """查看目录的树形结构

        Args:
            path: 目标目录路径，默认为工作目录
            max_depth: 递归深度限制，默认为 3
        """
        depth = max_depth if max_depth is not None else 3
        target = _resolve_path(path)
        guard.check_read(target)

        if not target.exists():
            return f"⚠️ 目录不存在: '{path}'"
        if not target.is_dir():
            return f"⚠️ 不是目录: '{path}'"

        stats = {"dirs": 0, "files": 0}
        lines = [f"{target.name}/"]
        lines.extend(_build_tree(target, "", depth, 1, stats))
        lines.append("")
        lines.append(f"{stats['dirs']} 个目录，{stats['files']} 个文件")

        return "\n".join(lines)

    # ... 返回列表增加 file_tree ...
    return [
        _build_tool(file_read),
        _build_tool(file_search),
        _build_tool(file_grep),
        _build_tool(file_edit),
        _build_tool(file_write),
        _build_tool(file_tree),
    ]
```

**Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py::TestFileTree -v`
Expected: 全部 PASS

**Step 5: 确认现有测试不被破坏**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py -v`
Expected: 全部 PASS（含新旧测试）

注意：现有 fixture `tools` 不使用 cwd，`create_file_tools(guard)` 的 cwd 默认 None 保持向后兼容。但 `test_returns_five_tools` 和 `test_tool_names` 需要更新为 7 个工具和新的工具名集合——但只在使用 cwd 时才有 7 个。不带 cwd 时仍然返回 5+1（file_tree 也应包含，因为它是通用工具）。需要决策：file_tree 是否始终包含。

**设计决策**：`file_tree` 和 `file_manage` 始终包含（只要调用了 `create_file_tools`），不依赖 cwd。不带 cwd 时使用第一个可读路径作为默认路径。因此现有测试 `test_returns_five_tools` 需改为 7，`test_tool_names` 需增加新工具名。

**Step 6: 更新现有工具数量测试**

修改 `tests/test_file_tools.py` 中的 `TestCreateFileTools`：

```python
class TestCreateFileTools:
    """create_file_tools 工厂函数测试"""

    def test_returns_seven_tools(self, guard):
        """应返回 7 个工具"""
        tools = create_file_tools(guard)
        assert len(tools) == 7
        assert all(isinstance(t, Tool) for t in tools)

    def test_tool_names(self, guard):
        """工具名称应正确"""
        names = {t.name for t in create_file_tools(guard)}
        assert names == {
            "file_read", "file_search", "file_grep",
            "file_edit", "file_write", "file_tree", "file_manage",
        }
```

**Step 7: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat(file_tools): 实现 file_tree 目录树查看工具"
```

---

### Task 3: 实现 file_manage 工具

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Test: `tests/test_file_tools.py`

**Step 1: 编写失败测试**

在 `tests/test_file_tools.py` 末尾新增 `TestFileManage` 类：

```python
class TestFileManage:
    """file_manage 工具测试"""

    async def test_mkdir_creates_directory(self, cwd_tools):
        """mkdir 应创建目录（含父目录）"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [{"action": "mkdir", "path": "a/b/c"}],
        })
        assert (cwd / "a" / "b" / "c").is_dir()
        assert "成功" in result

    async def test_delete_file(self, cwd_tools):
        """delete 应删除文件"""
        tools, cwd = cwd_tools
        f = cwd / "to_delete.txt"
        f.write_text("bye")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "delete", "path": "to_delete.txt"}],
        })
        assert not f.exists()
        assert "成功" in result

    async def test_delete_empty_dir(self, cwd_tools):
        """delete 应能删除空目录"""
        tools, cwd = cwd_tools
        d = cwd / "empty_dir"
        d.mkdir()
        result = await tools["file_manage"].execute({
            "operations": [{"action": "delete", "path": "empty_dir"}],
        })
        assert not d.exists()

    async def test_delete_nonempty_dir_without_recursive_fails(self, cwd_tools):
        """delete 非空目录且无 recursive 应失败"""
        tools, cwd = cwd_tools
        d = cwd / "nonempty"
        d.mkdir()
        (d / "file.txt").write_text("content")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "delete", "path": "nonempty"}],
        })
        assert d.exists()  # 目录仍在
        assert "失败" in result

    async def test_delete_nonempty_dir_with_recursive(self, cwd_tools):
        """delete 非空目录且 recursive=True 应成功"""
        tools, cwd = cwd_tools
        d = cwd / "nonempty"
        d.mkdir()
        (d / "file.txt").write_text("content")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "delete", "path": "nonempty", "recursive": True}],
        })
        assert not d.exists()
        assert "成功" in result

    async def test_move_file(self, cwd_tools):
        """move 应移动文件"""
        tools, cwd = cwd_tools
        src = cwd / "src.txt"
        src.write_text("hello")
        (cwd / "dest").mkdir()
        result = await tools["file_manage"].execute({
            "operations": [{"action": "move", "source": "src.txt", "destination": "dest/src.txt"}],
        })
        assert not src.exists()
        assert (cwd / "dest" / "src.txt").read_text() == "hello"

    async def test_copy_file(self, cwd_tools):
        """copy 应复制文件"""
        tools, cwd = cwd_tools
        src = cwd / "original.txt"
        src.write_text("data")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "copy", "source": "original.txt", "destination": "copy.txt"}],
        })
        assert src.exists()  # 原文件仍在
        assert (cwd / "copy.txt").read_text() == "data"

    async def test_copy_directory(self, cwd_tools):
        """copy 应递归复制目录"""
        tools, cwd = cwd_tools
        src = cwd / "srcdir"
        src.mkdir()
        (src / "a.txt").write_text("a")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "copy", "source": "srcdir", "destination": "dstdir"}],
        })
        assert (cwd / "dstdir" / "a.txt").read_text() == "a"

    async def test_rename_file(self, cwd_tools):
        """rename 应重命名文件"""
        tools, cwd = cwd_tools
        src = cwd / "old_name.txt"
        src.write_text("content")
        result = await tools["file_manage"].execute({
            "operations": [{"action": "rename", "source": "old_name.txt", "destination": "new_name.txt"}],
        })
        assert not src.exists()
        assert (cwd / "new_name.txt").read_text() == "content"

    async def test_serial_operations(self, cwd_tools):
        """串行操作应按顺序执行"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [
                {"action": "mkdir", "path": "project/src"},
                {"action": "copy", "source": ".", "destination": "project/backup"},
            ],
        })
        assert (cwd / "project" / "src").is_dir()
        assert "全部" in result and "成功" in result

    async def test_stop_on_failure(self, cwd_tools):
        """失败应停止后续操作"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [
                {"action": "delete", "path": "nonexist.txt"},
                {"action": "mkdir", "path": "should_not_create"},
            ],
        })
        assert not (cwd / "should_not_create").exists()
        assert "中止" in result or "失败" in result
        assert "未执行" in result

    async def test_sandbox_denied(self, cwd_tools):
        """沙箱外操作应被拒绝"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [{"action": "mkdir", "path": "/etc/evil"}],
        })
        assert "权限" in result or "沙箱" in result or "失败" in result

    async def test_invalid_action(self, cwd_tools):
        """无效 action 应返回错误"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [{"action": "invalid_action", "path": "test"}],
        })
        assert "失败" in result or "不支持" in result

    async def test_progress_report(self, cwd_tools):
        """应逐步报告执行进度"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [
                {"action": "mkdir", "path": "dir1"},
                {"action": "mkdir", "path": "dir2"},
            ],
        })
        assert "[1/2]" in result
        assert "[2/2]" in result

    async def test_move_sandbox_check(self, cwd_tools):
        """move 的 source 需要 read 权限，destination 需要 write 权限"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [{"action": "move", "source": "/etc/passwd", "destination": "stolen.txt"}],
        })
        assert "权限" in result or "沙箱" in result or "失败" in result

    async def test_rename_sandbox_check(self, cwd_tools):
        """rename 的 source 和 destination 都需要 write 权限"""
        tools, cwd = cwd_tools
        result = await tools["file_manage"].execute({
            "operations": [{"action": "rename", "source": "/etc/hosts", "destination": "/etc/hosts.bak"}],
        })
        assert "权限" in result or "沙箱" in result or "失败" in result
```

**Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py::TestFileManage -v`
Expected: 全部 FAIL

**Step 3: 实现 file_manage 工具**

在 `src/pure_agent_loop/file_tools.py` 的 `create_file_tools` 函数内，`file_tree` 之后添加：

```python
    def file_manage(operations: list[dict]) -> str:
        """文件管理操作管道

        支持串行执行多个文件操作，操作按顺序执行，失败即停止。
        路径支持相对路径（基于工作目录解析）。

        Args:
            operations: 操作列表，每个操作是一个字典，包含 action 和对应参数。
                支持的 action：
                - mkdir: 创建目录（含父目录）。参数：path
                - delete: 删除文件或目录。参数：path, recursive（可选，默认 false）
                - move: 移动文件/目录。参数：source, destination
                - copy: 复制文件/目录（目录递归复制）。参数：source, destination
                - rename: 重命名文件/目录。参数：source, destination
        """
        import shutil

        total = len(operations)
        results = []

        for i, op in enumerate(operations, 1):
            action = op.get("action", "")
            step_prefix = f"[{i}/{total}]"

            try:
                if action == "mkdir":
                    target = _resolve_path(op.get("path"))
                    guard.check_write(target)
                    target.mkdir(parents=True, exist_ok=True)
                    results.append(f"{step_prefix} mkdir {op.get('path')} → 成功")

                elif action == "delete":
                    target = _resolve_path(op.get("path"))
                    guard.check_write(target)
                    if not target.exists():
                        raise FileNotFoundError(f"路径不存在: {op.get('path')}")
                    if target.is_dir():
                        if any(target.iterdir()):
                            if not op.get("recursive", False):
                                raise OSError(f"目录非空，需设置 recursive=true: {op.get('path')}")
                            shutil.rmtree(target)
                        else:
                            target.rmdir()
                    else:
                        target.unlink()
                    results.append(f"{step_prefix} delete {op.get('path')} → 成功")

                elif action == "move":
                    src = _resolve_path(op.get("source"))
                    dst = _resolve_path(op.get("destination"))
                    guard.check_read(src)
                    guard.check_write(dst)
                    shutil.move(str(src), str(dst))
                    results.append(
                        f"{step_prefix} move {op.get('source')} → {op.get('destination')} → 成功"
                    )

                elif action == "copy":
                    src = _resolve_path(op.get("source"))
                    dst = _resolve_path(op.get("destination"))
                    guard.check_read(src)
                    guard.check_write(dst)
                    if src.is_dir():
                        shutil.copytree(str(src), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))
                    results.append(
                        f"{step_prefix} copy {op.get('source')} → {op.get('destination')} → 成功"
                    )

                elif action == "rename":
                    src = _resolve_path(op.get("source"))
                    dst = _resolve_path(op.get("destination"))
                    guard.check_write(src)
                    guard.check_write(dst)
                    src.rename(dst)
                    results.append(
                        f"{step_prefix} rename {op.get('source')} → {op.get('destination')} → 成功"
                    )

                else:
                    raise ValueError(f"不支持的操作: {action}")

            except Exception as e:
                results.append(f"{step_prefix} {action} → 失败: {e}")
                remaining = total - i
                if remaining > 0:
                    results.append(f"操作在第 {i} 步中止，后续 {remaining} 个操作未执行。")
                return "\n".join(results)

        results.append(f"\n全部 {total} 个操作执行成功。")
        return "\n".join(results)
```

返回列表增加 `file_manage`：

```python
    return [
        _build_tool(file_read),
        _build_tool(file_search),
        _build_tool(file_grep),
        _build_tool(file_edit),
        _build_tool(file_write),
        _build_tool(file_tree),
        _build_tool(file_manage),
    ]
```

注意：`file_manage` 的 `operations` 参数类型是 `list[dict]`，`_build_tool` 从类型注解提取 JSON Schema 时，`list[dict]` 会生成 `{"type": "array", "items": {"type": "object"}}`。LLM 需要知道 operations 内部结构，这通过 docstring 的 Args 描述传达。

**Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py::TestFileManage -v`
Expected: 全部 PASS

**Step 5: 运行全部文件工具测试**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py -v`
Expected: 全部 PASS

**Step 6: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat(file_tools): 实现 file_manage 操作管道工具"
```

---

### Task 4: Agent 集成（cwd 传递 + 提示词注入）

**Files:**
- Modify: `src/pure_agent_loop/agent.py:125-135`
- Modify: `src/pure_agent_loop/prompts.py`
- Test: `tests/test_agent.py`（如果有相关集成测试）

**Step 1: 编写失败测试**

在 `tests/` 下验证 Agent 集成（如果已有相关测试结构，在适当位置添加）：

```python
# tests/test_agent.py 或新建 tests/test_agent_sandbox.py

async def test_agent_sandbox_cwd_injects_prompt():
    """Agent 使用 sandbox cwd 时应在系统提示词中包含工作目录信息"""
    from unittest.mock import AsyncMock, MagicMock
    from pure_agent_loop import Agent, Sandbox
    from pure_agent_loop.llm.base import BaseLLMClient
    from pure_agent_loop.llm.types import LLMResponse, TokenUsage

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="done",
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    ))

    agent = Agent(
        llm=mock_llm,
        sandbox=Sandbox(cwd="/tmp/test_workspace"),
    )
    assert "工作目录" in agent._system_prompt
    assert "/tmp/test_workspace" in agent._system_prompt or "test_workspace" in agent._system_prompt
```

**Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && pytest tests/test_agent.py::test_agent_sandbox_cwd_injects_prompt -v`
Expected: FAIL

**Step 3: 修改 agent.py 和 prompts.py**

修改 `src/pure_agent_loop/agent.py` 第 125-135 行区域：

```python
        # 注册文件工具（如果配置了沙箱）
        sandbox_prompt = ""
        if sandbox:
            file_guard = SandboxGuard(sandbox)
            file_tools = create_file_tools(file_guard, cwd=sandbox.cwd)
            self._tool_registry.register_many(file_tools)
            if sandbox.cwd:
                sandbox_prompt = f"\n\n你的工作目录是：{sandbox.cwd}\n所有相对路径都基于此目录解析。"

        # 构建完整系统提示词
        self._system_prompt = build_system_prompt(
            name=name,
            user_prompt=system_prompt,
        ) + sandbox_prompt
```

**Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && pytest tests/test_agent.py -v`
Expected: 全部 PASS

**Step 5: 运行全部测试套件**

Run: `source venv/bin/activate && pytest -v`
Expected: 全部 PASS

**Step 6: 提交**

```bash
git add src/pure_agent_loop/agent.py tests/test_agent.py
git commit -m "feat(agent): 集成 sandbox cwd，工作目录注入系统提示词"
```

---

### Task 5: 更新现有工具的路径解析逻辑

**Files:**
- Modify: `src/pure_agent_loop/file_tools.py`
- Test: `tests/test_file_tools.py`

**Step 1: 编写测试验证相对路径基于 cwd 解析**

```python
# 在 tests/test_file_tools.py 中添加

class TestRelativePathResolution:
    """测试文件工具的相对路径解析"""

    async def test_file_read_relative_path(self, cwd_tools):
        """file_read 应支持相对路径（基于 cwd）"""
        tools, cwd = cwd_tools
        (cwd / "hello.txt").write_text("hello world")
        result = await tools["file_read"].execute({"file_path": "hello.txt"})
        assert "hello world" in result

    async def test_file_write_relative_path(self, cwd_tools):
        """file_write 应支持相对路径（基于 cwd）"""
        tools, cwd = cwd_tools
        result = await tools["file_write"].execute({
            "file_path": "new_file.txt",
            "content": "content",
        })
        assert (cwd / "new_file.txt").read_text() == "content"

    async def test_file_edit_relative_path(self, cwd_tools):
        """file_edit 应支持相对路径（基于 cwd）"""
        tools, cwd = cwd_tools
        (cwd / "edit_me.txt").write_text("old text")
        result = await tools["file_edit"].execute({
            "file_path": "edit_me.txt",
            "old_string": "old text",
            "new_string": "new text",
        })
        assert (cwd / "edit_me.txt").read_text() == "new text"

    async def test_file_search_relative_path(self, cwd_tools):
        """file_search 的 path 参数应支持相对路径"""
        tools, cwd = cwd_tools
        sub = cwd / "subdir"
        sub.mkdir()
        (sub / "found.py").write_text("pass")
        result = await tools["file_search"].execute({
            "pattern": "*.py", "path": "subdir",
        })
        assert "found.py" in result

    async def test_file_grep_relative_path(self, cwd_tools):
        """file_grep 的 path 参数应支持相对路径"""
        tools, cwd = cwd_tools
        sub = cwd / "subdir"
        sub.mkdir()
        (sub / "code.py").write_text("special_string")
        result = await tools["file_grep"].execute({
            "pattern": "special_string", "path": "subdir",
        })
        assert "code.py" in result
```

**Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py::TestRelativePathResolution -v`
Expected: FAIL（现有工具使用 `Path(file_path).resolve()` 而非通过 `_resolve_path`）

**Step 3: 更新现有 5 个工具的路径解析**

修改 `file_read`、`file_search`、`file_grep`、`file_edit`、`file_write` 中的路径解析逻辑，将 `Path(file_path).resolve()` 替换为 `_resolve_path(file_path)` 或 `_resolve_path(path)`。

关键改动点：
- `file_read`: `path = Path(file_path).resolve()` → `path = _resolve_path(file_path)`
- `file_search`: `search_path = Path(path).resolve()` → `search_path = _resolve_path(path)`
- `file_grep`: `search_path = Path(path).resolve()` → `search_path = _resolve_path(path)`
- `file_edit`: `path = Path(file_path).resolve()` → `path = _resolve_path(file_path)`
- `file_write`: `path = Path(file_path).resolve()` → `path = _resolve_path(file_path)`

**Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && pytest tests/test_file_tools.py -v`
Expected: 全部 PASS（新旧测试）

**Step 5: 提交**

```bash
git add src/pure_agent_loop/file_tools.py tests/test_file_tools.py
git commit -m "feat(file_tools): 所有文件工具支持基于 cwd 的相对路径解析"
```

---

### Task 6: 运行完整测试套件并确认

**Files:**
- 无新文件

**Step 1: 运行全部测试**

Run: `source venv/bin/activate && pytest -v`
Expected: 全部 PASS

**Step 2: 运行覆盖率报告**

Run: `source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing`
Expected: 新增代码有测试覆盖

**Step 3: 最终提交（如有遗漏修复）**

如果有测试失败，修复后提交。

---

### 任务依赖关系

```
Task 1 (Sandbox cwd)
  ↓
Task 2 (file_tree) ← 依赖 Task 1 的 cwd
  ↓
Task 3 (file_manage) ← 依赖 Task 2 的 create_file_tools 签名
  ↓
Task 4 (Agent 集成) ← 依赖 Task 1-3
  ↓
Task 5 (路径解析统一) ← 依赖 Task 2 的 _resolve_path
  ↓
Task 6 (全量验证)
```
