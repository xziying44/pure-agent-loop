# 工具参数自适应转换增强实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标**：增强 `_coerce_arguments()` 函数的类型转换能力，支持更宽松的布尔值和数字格式。

**架构**：仅修改 `src/pure_agent_loop/tool.py` 中的 `_coerce_arguments()` 函数（第26-63行），增强布尔值和数字的转换逻辑。完全向后兼容，不改变任何公开 API。

**技术栈**：Python 3.10+, pytest

---

## Task 1: 布尔值转换增强

**文件**：
- 修改：`src/pure_agent_loop/tool.py:54-55`
- 测试：`tests/test_tool.py`

### Step 1: 编写布尔值转换的失败测试

在 `tests/test_tool.py` 末尾添加新的测试类：

```python
class TestCoerceArgumentsEnhanced:
    """参数转换增强测试"""

    def test_boolean_conversion_case_insensitive(self):
        """布尔值转换应该大小写不敏感"""
        properties = {"flag": {"type": "boolean"}}

        # 真值测试
        assert _coerce_arguments({"flag": "True"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "TRUE"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "true"}, properties)["flag"] is True

    def test_boolean_conversion_extended_formats(self):
        """布尔值转换应该支持更多格式"""
        properties = {"flag": {"type": "boolean"}}

        # 真值：yes, on, t, y
        assert _coerce_arguments({"flag": "yes"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "YES"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "on"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "ON"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "t"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "y"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "1"}, properties)["flag"] is True

        # 假值：其他所有字符串
        assert _coerce_arguments({"flag": "false"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "no"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "off"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "0"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "random"}, properties)["flag"] is False

    def test_boolean_conversion_strips_whitespace(self):
        """布尔值转换应该去除首尾空格"""
        properties = {"flag": {"type": "boolean"}}

        assert _coerce_arguments({"flag": " true "}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "  yes  "}, properties)["flag"] is True
        assert _coerce_arguments({"flag": " false "}, properties)["flag"] is False
```

### Step 2: 运行测试验证失败

运行命令：
```bash
source venv/bin/activate
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced::test_boolean_conversion_case_insensitive -v
```

预期输出：`FAILED` - 当前实现不支持大小写不敏感

### Step 3: 实现布尔值转换增强

修改 `src/pure_agent_loop/tool.py` 第54-55行：

```python
elif expected_type == "boolean":
    normalized = value.lower().strip()
    coerced[key] = normalized in ("true", "1", "yes", "on", "t", "y")
```

### Step 4: 运行测试验证通过

运行命令：
```bash
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced -v
```

预期输出：所有布尔值测试 `PASSED`

### Step 5: 提交布尔值转换增强

```bash
git add src/pure_agent_loop/tool.py tests/test_tool.py
git commit -m "feat(tool): 增强布尔值参数转换

- 支持大小写不敏感（True/TRUE/true）
- 支持更多真值格式（yes/on/t/y）
- 自动去除首尾空格

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: 数字转换增强

**文件**：
- 修改：`src/pure_agent_loop/tool.py:50-53`
- 测试：`tests/test_tool.py`

### Step 1: 编写数字转换的失败测试

在 `tests/test_tool.py` 的 `TestCoerceArgumentsEnhanced` 类中添加：

```python
def test_integer_conversion_with_separators(self):
    """整数转换应该支持千分位分隔符和下划线"""
    properties = {"count": {"type": "integer"}}

    # 千分位分隔符
    assert _coerce_arguments({"count": "1,000"}, properties)["count"] == 1000
    assert _coerce_arguments({"count": "1,234,567"}, properties)["count"] == 1234567

    # Python 风格下划线
    assert _coerce_arguments({"count": "1_000"}, properties)["count"] == 1000
    assert _coerce_arguments({"count": "1_234_567"}, properties)["count"] == 1234567

    # 混合使用（虽然不推荐，但应该能处理）
    assert _coerce_arguments({"count": "1,000_000"}, properties)["count"] == 1000000

def test_integer_conversion_strips_whitespace(self):
    """整数转换应该去除首尾空格"""
    properties = {"count": {"type": "integer"}}

    assert _coerce_arguments({"count": " 123 "}, properties)["count"] == 123
    assert _coerce_arguments({"count": "  456  "}, properties)["count"] == 456

def test_float_conversion_with_separators(self):
    """浮点数转换应该支持千分位分隔符"""
    properties = {"price": {"type": "number"}}

    # 千分位分隔符
    assert _coerce_arguments({"price": "1,234.56"}, properties)["price"] == 1234.56
    assert _coerce_arguments({"price": "1_234.56"}, properties)["price"] == 1234.56

    # 科学计数法（原生支持）
    assert _coerce_arguments({"price": "1.23e5"}, properties)["price"] == 123000.0
    assert _coerce_arguments({"price": "1.23E5"}, properties)["price"] == 123000.0

def test_float_conversion_strips_whitespace(self):
    """浮点数转换应该去除首尾空格"""
    properties = {"price": {"type": "number"}}

    assert _coerce_arguments({"price": " 1.23 "}, properties)["price"] == 1.23
```

### Step 2: 运行测试验证失败

运行命令：
```bash
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced::test_integer_conversion_with_separators -v
```

预期输出：`FAILED` - ValueError: invalid literal for int()

### Step 3: 实现数字转换增强

修改 `src/pure_agent_loop/tool.py` 第50-53行：

```python
if expected_type == "integer":
    cleaned = value.replace(",", "").replace("_", "").strip()
    coerced[key] = int(cleaned)
elif expected_type == "number":
    cleaned = value.replace(",", "").replace("_", "").strip()
    coerced[key] = float(cleaned)
```

### Step 4: 运行测试验证通过

运行命令：
```bash
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced -v
```

预期输出：所有测试 `PASSED`

### Step 5: 提交数字转换增强

```bash
git add src/pure_agent_loop/tool.py tests/test_tool.py
git commit -m "feat(tool): 增强数字参数转换

- 支持千分位分隔符（1,000）
- 支持 Python 风格下划线（1_000）
- 自动去除首尾空格

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: 向后兼容性测试

**文件**：
- 测试：`tests/test_tool.py`

### Step 1: 编写向后兼容性测试

在 `tests/test_tool.py` 的 `TestCoerceArgumentsEnhanced` 类中添加：

```python
def test_backward_compatibility_correct_types(self):
    """已经是正确类型的参数不应该被转换"""
    properties = {
        "count": {"type": "integer"},
        "flag": {"type": "boolean"},
        "price": {"type": "number"},
    }

    args = {"count": 123, "flag": True, "price": 1.23}
    result = _coerce_arguments(args, properties)

    assert result["count"] == 123
    assert result["flag"] is True
    assert result["price"] == 1.23

def test_backward_compatibility_none_values(self):
    """None 值应该保持 None"""
    properties = {"count": {"type": "integer"}}

    result = _coerce_arguments({"count": None}, properties)
    assert result["count"] is None

def test_backward_compatibility_conversion_failure(self):
    """转换失败时应该保留原值"""
    properties = {"count": {"type": "integer"}}

    # 无法转换的字符串应该保留原值
    result = _coerce_arguments({"count": "abc"}, properties)
    assert result["count"] == "abc"

def test_backward_compatibility_unknown_parameters(self):
    """不在 schema 中的参数应该原样保留"""
    properties = {"count": {"type": "integer"}}

    result = _coerce_arguments({"count": "123", "unknown": "value"}, properties)
    assert result["count"] == 123
    assert result["unknown"] == "value"
```

### Step 2: 运行测试验证通过

运行命令：
```bash
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced::test_backward_compatibility -v
```

预期输出：所有向后兼容性测试 `PASSED`

### Step 3: 运行完整测试套件

运行命令：
```bash
pytest tests/test_tool.py -v
```

预期输出：所有测试 `PASSED`（包括原有测试和新增测试）

### Step 4: 提交向后兼容性测试

```bash
git add tests/test_tool.py
git commit -m "test(tool): 添加参数转换向后兼容性测试

确保增强不影响现有行为：
- 正确类型的参数不被转换
- None 值保持 None
- 转换失败时保留原值
- 未知参数原样保留

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 集成测试

**文件**：
- 测试：`tests/test_tool.py`

### Step 1: 编写端到端集成测试

在 `tests/test_tool.py` 的 `TestCoerceArgumentsEnhanced` 类中添加：

```python
async def test_integration_with_tool_execute(self):
    """集成测试：通过 Tool.execute() 验证参数转换"""

    @tool
    def process_data(count: int, enabled: bool, price: float) -> str:
        """处理数据

        Args:
            count: 数量
            enabled: 是否启用
            price: 价格
        """
        return f"count={count}, enabled={enabled}, price={price}"

    # 测试各种格式的参数
    result = await process_data.execute({
        "count": "1,000",
        "enabled": "YES",
        "price": "1_234.56"
    })

    assert result == "count=1000, enabled=True, price=1234.56"
```

### Step 2: 运行集成测试

运行命令：
```bash
pytest tests/test_tool.py::TestCoerceArgumentsEnhanced::test_integration_with_tool_execute -v
```

预期输出：`PASSED`

### Step 3: 运行完整测试套件（含覆盖率）

运行命令：
```bash
pytest tests/test_tool.py --cov=pure_agent_loop.tool --cov-report=term-missing -v
```

预期输出：
- 所有测试 `PASSED`
- `_coerce_arguments` 函数覆盖率 100%

### Step 4: 提交集成测试

```bash
git add tests/test_tool.py
git commit -m "test(tool): 添加参数转换端到端集成测试

验证增强后的参数转换在实际工具调用中正常工作

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 验收标准

完成后应满足：

1. ✅ 所有测试通过（`pytest tests/test_tool.py -v`）
2. ✅ 代码覆盖率不低于原有水平
3. ✅ 布尔值支持：`true/yes/on/1/t/y`（大小写不敏感）
4. ✅ 数字支持：千分位分隔符、下划线、科学计数法
5. ✅ 向后兼容：不影响现有功能
6. ✅ 4 个独立提交，每个提交可独立运行测试

## 预计时间

- Task 1: 15 分钟
- Task 2: 15 分钟
- Task 3: 10 分钟
- Task 4: 10 分钟
- **总计**: 约 50 分钟
