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
                {"action": "mkdir", "path": "project/tests"},
            ],
        })
        assert (cwd / "project" / "src").is_dir()
        assert (cwd / "project" / "tests").is_dir()
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
