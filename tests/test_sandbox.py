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


class TestSandboxCwd:
    """Sandbox cwd 字段测试"""

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
