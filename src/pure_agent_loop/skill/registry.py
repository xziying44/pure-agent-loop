"""Skill 注册表"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .parser import SkillParser
from .scanner import SkillScanner
from .types import SkillContent, SkillInfo

logger = logging.getLogger(__name__)


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

            for file_path in paths:
                info = SkillParser.parse_frontmatter(file_path)
                if not info:
                    continue

                if info.name in self._skills:
                    existing = self._skills[info.name]
                    logger.warning(
                        "重复的 Skill 名称被忽略: %s (已存在: %s, 被忽略: %s)",
                        info.name,
                        existing.location,
                        file_path,
                    )
                    continue

                self._skills[info.name] = info

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
        return sorted(self._skills.values(), key=lambda s: s.name)

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

    def _normalize_relative_path(self, user_path: str) -> Path:
        """将用户输入路径规范化为相对路径

        目的：
        - 避免用户传入绝对路径直接跳出 base_dir
        - 统一后续的 resolve + 越界检查逻辑
        """
        p = Path(user_path)
        if p.is_absolute():
            # 去掉根路径，使其变为相对路径，例如 /etc/passwd -> etc/passwd
            p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        return p

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

        rel = self._normalize_relative_path(file_path)
        target_path = (info.base_dir / rel).resolve()

        if not self._is_path_safe(target_path, info.base_dir):
            return f"错误：路径 '{file_path}' 越界，禁止访问 Skill 目录外的文件"

        if not target_path.exists():
            return f"错误：文件 '{file_path}' 不存在"
        if not target_path.is_file():
            return f"错误：'{file_path}' 不是文件"

        try:
            return target_path.read_text(encoding="utf-8")
        except Exception:
            logger.debug("读取 Skill 文件失败: %s", target_path, exc_info=True)
            return "错误：读取文件失败"

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
        import re
        import subprocess
        import sys

        info = self._skills.get(skill_name)
        if not info:
            return f"错误：Skill '{skill_name}' 不存在"

        # 禁止包含路径，避免 ../ 或子目录形式
        if Path(script_name).name != script_name:
            return "错误：脚本名不合法，禁止包含路径"

        script = next((s for s in info.scripts if s.name == script_name), None)
        if not script:
            declared = [s.name for s in info.scripts]
            return (
                f"错误：脚本 '{script_name}' 未在 Skill '{skill_name}' 中声明。\n"
                f"已声明的脚本：{declared}"
            )

        script_path = (info.base_dir / "scripts" / script_name).resolve()
        if not self._is_path_safe(script_path, info.base_dir):
            return "错误：脚本路径越界，禁止执行"

        if not script_path.exists():
            return f"错误：脚本文件 '{script_path}' 不存在"
        if not script_path.is_file():
            return f"错误：脚本 '{script_name}' 不是文件"

        cmd: list[str] = [sys.executable, str(script_path)]
        if args:
            for key, value in args.items():
                if not re.match(r"^[a-zA-Z0-9_-]+$", str(key)):
                    return f"错误：参数名不合法: {key}"
                cmd.extend([f"--{key}", str(value)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(info.base_dir),
            )
        except subprocess.TimeoutExpired:
            return "错误：脚本执行超时（60秒）"
        except Exception:
            logger.debug("执行 Skill 脚本失败: %s", script_path, exc_info=True)
            return "错误：脚本执行失败"

        output = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            if stderr:
                return f"{output}\n[stderr]: {stderr}".strip()
            return output or "错误：脚本执行失败（无输出）"

        return output or "(脚本执行完成，无输出)"
