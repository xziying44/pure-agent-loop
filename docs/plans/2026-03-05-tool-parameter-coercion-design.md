# 工具参数自适应转换增强设计

**日期**：2026-03-05
**状态**：已批准
**作者**：Claude (Kiro)

## 背景

当前 `pure-agent-loop` 的工具系统已支持基础的参数类型转换（字符串 → int/float/bool/array/object），但在实际使用中发现：

1. **布尔值识别过于严格**：仅支持 `"true"/"1"/"yes"`（小写），LLM 可能生成 `"True"/"on"/"yes"` 等变体
2. **数字格式容错不足**：不支持千分位分隔符（`"1,000"`）和 Python 风格下划线（`"1_000"`）

这些限制导致 LLM 生成的参数格式稍有偏差就会触发类型错误，影响智能体的鲁棒性。

## 目标

增强 `_coerce_arguments()` 函数的类型转换能力，使其对 LLM 生成的参数格式更加宽容，减少因格式问题导致的工具调用失败。

**非目标**：
- 不改动异常处理逻辑（保持现状）
- 不引入新的 API 或配置项
- 不改变转换失败时的行为（保留原值）

## 设计方案

### 架构

**改动范围**：仅修改 `src/pure_agent_loop/tool.py` 中的 `_coerce_arguments()` 函数。

**向后兼容性**：完全兼容，不改变任何公开 API。

### 参数转换增强

#### 1. 布尔值转换

**当前实现**：
```python
coerced[key] = value.lower() in ("true", "1", "yes")
```

**增强后**：
```python
normalized = value.lower().strip()
coerced[key] = normalized in ("true", "1", "yes", "on", "t", "y")
```

**支持的格式**：
- 真值：`"true"/"yes"/"on"/"1"/"t"/"y"`（大小写不敏感）
- 假值：其他所有字符串
- 自动去除首尾空格

#### 2. 数字转换

**当前实现**：
```python
coerced[key] = int(value)  # 或 float(value)
```

**增强后**：
```python
cleaned = value.replace(",", "").replace("_", "").strip()
coerced[key] = int(cleaned)  # 或 float(cleaned)
```

**支持的格式**：
- 千分位分隔符：`"1,000"` → 1000
- Python 风格下划线：`"1_000"` → 1000
- 科学计数法：`"1.23e5"` → 123000.0（float 原生支持）
- 自动去除首尾空格

#### 3. 转换失败策略

保持现状：转换失败时保留原值，让函数的类型检查机制报错。

**理由**：
- 保留原始错误信息，便于调试
- 避免静默失败导致的逻辑错误
- 让工具开发者能够看到真实的参数值

### 数据流

参数转换的执行流程保持不变：

```
LLM 返回 tool_calls
    ↓
Tool.execute(arguments: dict)
    ↓
_coerce_arguments(arguments, schema)  ← 增强点
    ↓
function(**coerced_args)
    ↓
返回结果或错误信息
```

**关键点**：
- 转换发生在函数调用前，对 LLM 和工具函数都是透明的
- 转换失败时保留原值，让函数的类型检查机制报错
- 不改变现有的错误处理流程

## 测试策略

在 `tests/test_tool.py` 中新增测试用例：

### 1. 布尔值转换测试

```python
# 真值测试
assert coerce("true") == True
assert coerce("True") == True
assert coerce("TRUE") == True
assert coerce("yes") == True
assert coerce("on") == True
assert coerce("1") == True
assert coerce("t") == True
assert coerce("y") == True
assert coerce(" true ") == True  # 去除空格

# 假值测试
assert coerce("false") == False
assert coerce("no") == False
assert coerce("off") == False
assert coerce("0") == False
```

### 2. 数字转换测试

```python
# 整数转换
assert coerce("1000") == 1000
assert coerce("1,000") == 1000
assert coerce("1_000") == 1000
assert coerce(" 123 ") == 123

# 浮点数转换
assert coerce("1.23") == 1.23
assert coerce("1,234.56") == 1234.56
assert coerce("1.23e5") == 123000.0
```

### 3. 转换失败测试

```python
# 转换失败时保留原值
assert coerce("abc", target_type="integer") == "abc"
```

### 4. 向后兼容测试

```python
# 已经是正确类型的参数不受影响
assert coerce(123) == 123
assert coerce(True) == True

# None 值保持 None
assert coerce(None) == None
```

## 实施计划

1. 修改 `_coerce_arguments()` 函数
2. 编写测试用例
3. 运行测试确保通过
4. 更新文档（如有必要）

## 风险与缓解

**风险**：布尔值转换可能过于宽容，导致意外的真值判断。

**缓解**：
- 仅支持明确的真值关键词（`"true"/"yes"/"on"/"1"/"t"/"y"`）
- 其他所有字符串都视为假值
- 在测试中覆盖边缘情况

## 未来扩展

如果需要更强的扩展性（例如支持日期时间、枚举类型等），可以考虑：
- 引入类型转换器注册机制
- 允许工具开发者自定义转换逻辑

但当前阶段遵循 YAGNI 原则，仅实现已知需求。
