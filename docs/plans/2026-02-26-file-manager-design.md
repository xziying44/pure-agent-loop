# 文件管理器设计方案

## 概述

为 pure-agent-loop 新增文件管理器功能，包含目录树查看工具和文件管理操作管道工具，同时增强 Sandbox 支持工作目录概念。所有工具均受沙盒权限控制。

## 变更范围

### 1. Sandbox 增加工作目录（cwd）

**文件：** `sandbox.py`

`Sandbox` 新增 `cwd` 字段，作为 Agent 的工作目录：

```python
@dataclass
class Sandbox:
    cwd: str | Path              # 工作目录，自动加入 write_paths
    read_paths: list[str | Path] = field(default_factory=list)
    write_paths: list[str | Path] = field(default_factory=list)

    def __post_init__(self):
        self.cwd = Path(self.cwd).resolve()
        self.write_paths = [Path(p).resolve() for p in self.write_paths]
        self.write_paths.append(self.cwd)  # cwd 自动拥有读写权限
        self.read_paths = [Path(p).resolve() for p in self.read_paths]
```

**行为：**
- `cwd` 自动加入 `write_paths`（隐含读写权限），用户无需重复配置
- 最简使用：`Sandbox(cwd="/path/to/project")`
- `cwd` 被注入到系统提示词中，Agent 知道默认工作目录
- 所有文件工具收到相对路径时，以 `cwd` 为基准解析

### 2. file_tree 工具

**文件：** `file_tools.py`

独立的目录树查看工具，语义清晰。

**签名：**
```python
file_tree(path: str | None, max_depth: int | None) -> str
```

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | `str \| None` | 否 | 目标目录路径，默认为 cwd |
| `max_depth` | `int \| None` | 否 | 递归深度限制，默认为 3 |

**输出格式：**
```
output/
├── styles/
│   ├── main.css
│   └── reset.css
├── index.html
└── app.js

3 个目录，4 个文件
```

**特性：**
- 受沙盒读权限控制
- 忽略 `.git`、`__pycache__`、`node_modules` 等（复用现有忽略列表）
- 超过深度限制的目录显示 `[...]` 标记

### 3. file_manage 工具

**文件：** `file_tools.py`

支持串行操作管道的文件管理工具，操作按顺序执行，失败即停止。

**签名：**
```python
file_manage(operations: list[dict]) -> str
```

**操作类型：**

| action | 必填参数 | 可选参数 | 说明 |
|--------|---------|---------|------|
| `mkdir` | `path` | - | 创建目录（含父目录，类似 `mkdir -p`） |
| `delete` | `path` | `recursive` (bool, 默认 false) | 删除文件或目录。非空目录需 `recursive=true` |
| `move` | `source`, `destination` | - | 移动文件/目录 |
| `copy` | `source`, `destination` | - | 复制文件/目录（目录递归复制） |
| `rename` | `source`, `destination` | - | 重命名文件/目录 |

**权限检查：**
- `mkdir`：`check_write(path)`
- `delete`：`check_write(path)`
- `move`：`check_read(source)` + `check_write(destination)`
- `copy`：`check_read(source)` + `check_write(destination)`
- `rename`：`check_write(source)` + `check_write(destination)`

**串行执行与失败处理：**

操作按数组顺序依次执行，任一操作失败则立即停止，后续操作不执行。

成功返回示例：
```
[1/3] mkdir output/styles → 成功
[2/3] copy src/main.css → output/styles/main.css → 成功
[3/3] delete temp.txt → 成功

全部 3 个操作执行成功。
```

失败返回示例：
```
[1/3] mkdir output/styles → 成功
[2/3] copy not_exist.css → output/styles/main.css → 失败: 源文件不存在
操作在第 2 步中止，后续 1 个操作未执行。
```

**JSON Schema：**
```json
{
  "type": "object",
  "properties": {
    "operations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "action": {"type": "string", "enum": ["mkdir", "delete", "move", "copy", "rename"]},
          "path": {"type": "string", "description": "目标路径（mkdir/delete 使用）"},
          "source": {"type": "string", "description": "源路径（move/copy/rename 使用）"},
          "destination": {"type": "string", "description": "目标路径（move/copy/rename 使用）"},
          "recursive": {"type": "boolean", "description": "递归删除（仅 delete 使用，默认 false）"}
        },
        "required": ["action"]
      }
    }
  },
  "required": ["operations"]
}
```

### 4. 集成改动

**agent.py：**
- `create_file_tools(guard, cwd)` 工厂函数增加 `cwd` 参数
- 返回的工具列表新增 `file_tree` 和 `file_manage`（5 → 7 个）
- 提示词注入工作目录信息

**prompts.py：**
- 系统提示词追加：`你的工作目录是：{cwd}\n所有相对路径都基于此目录解析。`

## 文件变更清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `sandbox.py` | 修改 | `Sandbox` 增加 `cwd` 字段，自动加入 `write_paths` |
| `file_tools.py` | 修改 | 新增 `file_tree` 和 `file_manage` 工具；`create_file_tools` 增加 `cwd` 参数 |
| `agent.py` | 修改 | 传递 `cwd` 给 `create_file_tools`，提示词注入工作目录 |
| `tests/test_file_tools.py` | 修改 | 新增 `file_tree` 和 `file_manage` 测试用例 |
| `tests/test_sandbox.py` | 修改 | 新增 `cwd` 自动加入权限的测试 |

## 测试要点

- **file_tree**：空目录、嵌套目录、depth 限制、忽略目录列表、权限拦截
- **file_manage**：单操作、串行多操作、失败即停止、各 action 类型、权限检查、相对路径基于 cwd 解析
- **Sandbox.cwd**：自动加入 write_paths、cwd 提示词注入
