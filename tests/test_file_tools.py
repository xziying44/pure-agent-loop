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
