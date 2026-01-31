# 工具类注册设计文档

> 日期：2026-01-31
> 状态：已批准
> 目标：支持以类的方式批量注册工具方法，自动扫描 `@tool` 标记的方法并以 `前缀_方法名` 格式注册

---

## 1. 背景与动机

当前工具系统仅支持以独立函数或字典的方式注册工具。当多个工具在逻辑上属于同一领域（如文件操作、数据库操作）时，缺乏分组和归属机制。

需要一种类级别的注册方式：
- 将相关工具组织在一个类中
- 自动扫描类中带 `@tool` 标记的方法
- 工具名称格式化为 `前缀_方法名`，确保归属明确

## 2. 核心设计

### 2.1 `@tool` 双模式

`@tool` 装饰器根据函数签名自动切换行为：

| 场景 | 第一个参数 | 行为 | 返回值 |
|------|-----------|------|--------|
| 独立函数 | 非 `self`/`cls` | 转换为 Tool 对象 | `Tool` |
| 类方法 | `self` 或 `cls` | 仅标记 `_tool_marker = True` | 原始函数 |

```python
# 独立函数 — 行为不变
@tool
def search(query: str) -> str:
    """搜索网页"""
    return f"结果: {query}"
# search 是 Tool 对象

# 类方法 — 仅标记
class FileManager:
    @tool
    def read_file(self, path: str) -> str:
        """读取文件"""
        return open(path).read()
# read_file 仍是普通方法，可正常调用
```

### 2.2 类扫描与注册

`ToolRegistry.register_class(obj)` 流程：

```
传入实例或类
    ↓
如果是类 → 无参实例化（失败则抛 TypeError）
    ↓
读取 tool_prefix 属性（无则用类名）
    ↓
遍历类属性，找到 _tool_marker = True 的方法
    ↓
对每个方法：
  1. 绑定到实例 → bound_method（消除 self）
  2. 解析 docstring + 类型注解（跳过 self/cls）
  3. 名称 = f"{prefix}_{method_name}"
  4. 创建 Tool 对象并注册
    ↓
如果无标记方法 → 发出 warnings.warn
```

### 2.3 名称规则

- 分隔符：下划线 `_`
- 前缀来源：类属性 `tool_prefix`（可选），未声明时使用类名
- 示例：`tool_prefix = "fm"` + `read_file` → `fm_read_file`
- 示例：无 prefix + `FileManager` + `read_file` → `FileManager_read_file`

### 2.4 注册 API

```python
# ToolRegistry 新增方法
registry.register_class(FileManager())      # 传实例
registry.register_class(FileManager)        # 传类，自动无参实例化

# register / register_many 自动路由
registry.register(FileManager())            # 检测为类实例 → 路由到 register_class
registry.register_many([search, FileManager()])  # 混合列表

# Agent 入口透传
agent = Agent(
    model="deepseek-chat",
    tools=[search, FileManager("/data")],
)
```

**识别逻辑**：在 `register()` 中，按以下顺序判断：
1. `isinstance(obj, Tool)` → 直接注册
2. `isinstance(obj, dict)` → 字典格式注册
3. `isinstance(obj, type)` → 类本身，路由到 `register_class`
4. 其他对象 → 检查是否有 `_tool_marker` 方法，有则路由到 `register_class`

## 3. 边界情况

### 3.1 名称冲突
静默覆盖（与现有 `register` 行为一致）。

### 3.2 无参实例化失败
```python
raise TypeError(
    f"无法自动实例化类 '{cls.__name__}'：构造函数需要参数。"
    f"请传入类的实例而非类本身。"
)
```

### 3.3 类中无 @tool 方法
```python
import warnings
warnings.warn(f"类 '{cls.__name__}' 中未发现任何 @tool 标记的方法")
```

### 3.4 静态方法和类方法
- `@staticmethod` + `@tool`：支持，无 `self`，直接按普通函数处理
- `@classmethod` + `@tool`：支持，跳过 `cls` 参数，绑定到类
- 装饰器顺序：`@tool` 必须在内层

```python
class MyTools:
    @staticmethod
    @tool
    def ping() -> str:
        """测试连通性"""
        return "pong"
```

## 4. 文件变更清单

| 文件 | 变更内容 |
|------|----------|
| `src/pure_agent_loop/tool.py` | `@tool` 双模式检测；新增 `register_class()`；`register()` / `register_many()` 增加路由 |
| `tests/test_tool.py` | 新增 `TestToolClassRegistration` 测试类 |

**不涉及**：`agent.py`、`loop.py`、`events.py` 等无需修改。

## 5. 完整用法示例

```python
from pure_agent_loop import Agent, tool

class FileManager:
    tool_prefix = "fm"

    def __init__(self, base_dir: str = "/tmp"):
        self.base_dir = base_dir

    @tool
    def read_file(self, path: str) -> str:
        """读取文件内容

        Args:
            path: 文件路径
        """
        return open(f"{self.base_dir}/{path}").read()

    @tool
    async def write_file(self, path: str, content: str) -> str:
        """写入文件内容

        Args:
            path: 文件路径
            content: 写入内容
        """
        with open(f"{self.base_dir}/{path}", "w") as f:
            f.write(content)
        return f"已写入 {path}"

    def _internal(self):
        """内部方法，不会被注册"""
        ...

@tool
def search(query: str) -> str:
    """搜索网页"""
    return f"结果: {query}"

agent = Agent(
    model="deepseek-chat",
    tools=[search, FileManager("/data")],
)
result = agent.run("读取 config.json 文件")
# 注册的工具：search, fm_read_file, fm_write_file
```
