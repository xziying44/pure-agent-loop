# 文件工具 + 沙箱系统设计

## 概述

为 pure-agent-loop 添加 5 个内置文件工具（file_read、file_search、file_grep、file_edit、file_write），配合沙箱权限系统控制文件访问范围。

## 需求

- 创建 Agent 时通过 `sandbox` 参数指定允许访问的路径
- 路径分为 `read_paths`（仅读）和 `write_paths`（可读写）
- 沙箱外路径严格拒绝，不可访问
- 文件工具自动注册，与用户自定义工具共存

## 使用示例

```python
from pure_agent_loop import Agent, Sandbox

agent = Agent(
    model="deepseek-chat",
    tools=[my_custom_tool],
    sandbox=Sandbox(
        read_paths=["/data/docs"],
        write_paths=["/workspace"]
    )
)
# file_read, file_search, file_grep, file_edit, file_write 自动注册
```

## 模块结构

```
pure_agent_loop/
├── sandbox.py              # Sandbox, SandboxGuard, SandboxViolationError
├── builtin_tools/
│   ├── __init__.py
│   ├── todo_tools.py       # 已有
│   └── file_tools.py       # 新增：5 个文件工具 + create_file_tools()
```

## 沙箱配置与路径验证

### Sandbox 数据类

```python
@dataclass
class Sandbox:
    read_paths: list[str | Path] = field(default_factory=list)
    write_paths: list[str | Path] = field(default_factory=list)
```

- `read_paths` — 仅允许读取的路径列表（目录或文件）
- `write_paths` — 允许读写的路径列表（隐含读权限）
- 所有路径在初始化时自动 `resolve()` 为绝对路径

### SandboxGuard 验证器

```python
class SandboxGuard:
    def __init__(self, sandbox: Sandbox): ...
    def check_read(self, path: str | Path) -> None:   # 失败抛 SandboxViolationError
    def check_write(self, path: str | Path) -> None:   # 失败抛 SandboxViolationError
    def is_readable(self, path: str | Path) -> bool:    # 不抛异常
```

路径检查：目标路径 `resolve()` 后，检查是否为任一沙箱路径的子路径（`Path.is_relative_to()`）。

### SandboxViolationError

```python
class SandboxViolationError(PureAgentLoopError):
    def __init__(self, path: str, operation: str, allowed_paths: list[str]): ...
```

## 文件工具 API

### file_read — 读取文件

```python
def file_read(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
```

- 权限：读
- 返回带行号的内容（`1\t内容` 格式）
- 默认 limit=2000，单行超 2000 字符截断
- 二进制文件拒绝（扩展名黑名单 + 空字节采样）
- 目录路径返回目录列表（仅沙箱内可见条目）

### file_search — glob 文件名搜索

```python
def file_search(pattern: str, path: str | None = None) -> str:
```

- 权限：读
- 基于 `pathlib.Path.glob()`
- 结果按修改时间降序，最多 100 个
- 自动跳过 .git、__pycache__、node_modules 等

### file_grep — 正则内容搜索

```python
def file_grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
```

- 权限：读
- 纯 Python（`re` + `pathlib`）
- 返回 `文件路径:行号: 匹配行内容`，最多 100 个匹配
- 自动跳过常见忽略目录

### file_edit — 精确字符串替换

```python
def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
```

- 权限：写
- 精确匹配，未找到返回错误
- 多处匹配且 replace_all=False 时报错
- 返回统一 diff 格式变更摘要

### file_write — 创建或重写文件

```python
def file_write(file_path: str, content: str) -> str:
```

- 权限：写
- 自动创建不存在的父目录
- 已有文件返回 diff，新文件标记为新建

## Agent 集成

### 新增参数

```python
class Agent:
    def __init__(self, ..., sandbox: Sandbox | None = None):
```

### 构造流程

```python
if sandbox:
    guard = SandboxGuard(sandbox)
    file_tools = create_file_tools(guard)
    self._tools.extend(file_tools)
```

- `sandbox=None` 时不注册文件工具（向后兼容）
- 文件工具追加到用户自定义工具之后

### 公开 API 新增导出

```python
from .sandbox import Sandbox, SandboxGuard, SandboxViolationError
```

## 错误处理

文件工具错误不中断循环，返回格式化字符串给 LLM：

```
⚠️ 权限不足: '/etc/passwd' 不在沙箱允许的读取路径内。允许的路径: ['/data/docs', '/workspace']
⚠️ 文件不存在: '/workspace/nonexistent.py'
⚠️ 未找到匹配内容，请检查 old_string 是否与文件内容完全一致
⚠️ 找到 3 处匹配，请提供更多上下文使匹配唯一，或设置 replace_all=True
⚠️ 无法读取二进制文件: '/workspace/image.png'
```

`SandboxViolationError` 仅框架内部使用，`Tool.execute()` 捕获后转为格式化字符串。

## 测试策略

| 文件 | 覆盖内容 |
|------|---------|
| `tests/test_sandbox.py` | Sandbox 配置、SandboxGuard 路径验证、边界情况 |
| `tests/test_file_tools.py` | 5 个工具功能测试、权限拒绝、边界情况 |

关键用例：
- 读取沙箱内/外文件
- 写入 read_paths（应拒绝）
- 写入 write_paths（应允许）
- glob/grep 结果自动过滤沙箱外路径
- 符号链接逃逸防护（resolve 后再检查）
- Agent 集成测试（sandbox → 工具自动注册）
- 全部使用 `tmp_path` fixture，不依赖真实文件系统

## 技术决策

- 底层实现：纯标准库（pathlib + re + fnmatch），零外部依赖
- 编辑策略：仅精确匹配，行为最可预测
- 集成方式：内置工具模式，Agent 参数自动注册
- 沙箱策略：严格拒绝，沙箱外路径完全不可访问
