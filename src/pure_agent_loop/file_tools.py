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

# 忽略的目录名
IGNORED_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".idea", ".vscode", ".eggs", "*.egg-info",
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


def create_file_tools(guard: SandboxGuard, cwd: Path | None = None) -> list[Tool]:
    """创建文件工具列表

    通过闭包捕获 SandboxGuard 实例，实现路径权限控制。

    Args:
        guard: 沙箱路径验证器
        cwd: 工作目录，用于解析相对路径和作为默认路径

    Returns:
        包含文件工具的列表
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

    def file_read(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        """读取文件内容

        Args:
            file_path: 文件路径（支持相对路径）
            offset: 起始行号（从 1 开始），不指定则从头读取
            limit: 最多读取的行数，默认 2000
        """
        path = _resolve_path(file_path)

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

    def file_search(pattern: str, path: str | None = None) -> str:
        """按文件名模式搜索文件

        Args:
            pattern: glob 模式，如 '**/*.py'
            path: 搜索起始目录，默认为工作目录或第一个可读路径
        """
        search_path = _resolve_path(path)
        guard.check_read(search_path)

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
            path: 搜索目录，默认为工作目录或第一个可读路径
            include: 文件名过滤模式，如 '*.py'
        """
        import re
        import fnmatch

        search_path = _resolve_path(path)
        guard.check_read(search_path)

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

    def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """精确替换文件中的文本

        Args:
            file_path: 文件路径
            old_string: 要替换的原始文本（必须精确匹配）
            new_string: 替换后的文本
            replace_all: 是否替换所有匹配项，默认仅替换第一个
        """
        import difflib

        path = _resolve_path(file_path)
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

    def file_write(file_path: str, content: str) -> str:
        """创建新文件或完全重写已有文件

        Args:
            file_path: 文件路径
            content: 完整的文件内容
        """
        import difflib

        path = _resolve_path(file_path)
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

    # 构建 Tool 对象列表
    from .tool import _build_tool
    return [
        _build_tool(file_read),
        _build_tool(file_search),
        _build_tool(file_grep),
        _build_tool(file_edit),
        _build_tool(file_write),
        _build_tool(file_tree),
        _build_tool(file_manage),
    ]
