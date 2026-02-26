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
