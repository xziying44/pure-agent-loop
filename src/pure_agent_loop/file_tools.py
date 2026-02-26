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
