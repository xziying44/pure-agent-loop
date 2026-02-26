# 系统提示词 + Todo + 循环体 + Skill 引导 全面重构设计

## 背景

参考 opencode 项目的成熟设计，重构 pure-agent-loop 的四个核心模块：
1. 系统提示词（prompts.py）
2. Todo 工具（builtin_tools.py）
3. 循环体（loop.py + limits.py）
4. Skill 引导（集成在 prompts.py 中）

opencode 偏向编程场景，pure-agent-loop 偏向通用场景。

## 设计约束

- **单一通用提示词**：不做模型级切换
- **Todo 四状态对齐 opencode**：pending / in_progress / completed / cancelled
- **温和引导风格**：参考 opencode 的 "frequently + proactively"
- **用户提示词仍通过 API 参数注入**：保持 `build_system_prompt(name, user_prompt)` 接口

## 模块 1：系统提示词重构

### 文件：`prompts.py`

### 接口不变

```python
def build_system_prompt(name: str = "智能助理", user_prompt: str = "") -> str
```

### 新提示词结构（6 段）

**1. 角色定义**
- "你是{name}，一个高效的智能助理"
- 通用场景定位（非编程专用）

**2. 专业客观性（参考 opencode `anthropic.txt`）**
- 优先技术准确性和真实性，而非迎合用户
- 不确定时先调查而非确认用户预设
- 客观指导和尊重的纠正比虚假同意更有价值

**3. 任务管理引导（opencode 风格温和引导）**
- "你可以使用 `todo_write` 工具来管理和规划任务，频繁使用确保任务追踪和进度可见"
- "对于规划任务和将大型复杂任务分解为更小步骤极为有用"
- 明确何时用 / 何时不用（精选 opencode todowrite.txt 的例子精华，适配通用场景）
- 四状态说明：pending → in_progress → completed / cancelled
- "一次只有一个 in_progress，完成后立即标记"

**4. Skill 系统引导（温和版）**
- "你有一个技能索引，包含可用技能的名称和描述"
- "在处理任务时，检查是否有匹配的技能"
- 简洁的使用 / 不使用指南
- 去掉所有"严禁"、"唯一起手式"、"综合研判"等强制用语

**5. 工具使用策略（新增，参考 opencode `codex_header.txt`）**
- 并行调用独立工具
- 优先使用专用工具而非通用方法

**6. 输出风格指南（新增，参考 opencode `anthropic.txt`）**
- 简洁、友好、协作语气
- 默认直接做工作而非问"要不要继续"
- 只在真正受阻时才提问
- 如果必须提问：先做所有非阻塞工作，然后问一个有针对性的问题
- 用户自定义指令注入区

### 关键变化

- 去掉"SOP 驱动"概念和所有强制协议语气
- 新增"专业客观性"和"输出风格指南"
- Skill 和 Todo 从强制协议改为 proactive 引导

## 模块 2：Todo 工具重构

### 文件：`builtin_tools.py`

### A. 状态扩展为四状态

```python
VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")
```

### B. 新增 priority 字段

```python
@dataclass
class TodoItem:
    content: str
    status: str = "pending"
    priority: str = "medium"  # high / medium / low
```

### C. 工具描述完全重写

参照 opencode `todowrite.txt`（167 行），翻译并适配为通用场景版本：
- 保留"何时用 / 何时不用"的详细指南和示例
- 保留"一次只有一个 in_progress"的规则
- 保留"立即标记完成，不要批量标记"的规则
- 示例改为通用场景（非编程专用）

### D. 保持 write-only 语义

每次替换整个列表，不引入增量操作。

### E. 事件产出不变

循环体中检测到 `todo_write` 调用后仍产出 `TODO_UPDATE` 事件。

## 模块 3：循环体重构

### 文件：`loop.py` + `limits.py`

### 核心设计：双模式限制体系

#### LoopLimits 新接口

```python
@dataclass
class LoopLimits:
    max_steps: int | None = None           # 步数上限，None=无限
    step_limit_mode: str = "soft"          # "soft" | "hard"
    max_tokens: int = 100_000              # token 硬限制（始终为硬）
    doom_loop_threshold: int = 3           # doom loop 检测阈值

    # 软限制：提醒提示词（注入后继续执行）
    soft_limit_prompt: str = "..."
    # 软限制：首次触发后每隔多少步再次提醒（默认=max_steps）
    soft_limit_interval: int | None = None

    # 硬限制：终止提示词（禁用工具，要求总结）
    hard_limit_prompt: str = "..."         # 对标 opencode max-steps.txt
```

#### 软限制模式（适合长任务 / 主循环）

- 达到 `max_steps` 后产出 `SOFT_LIMIT` 事件
- 注入 `soft_limit_prompt` 到消息历史，但不禁用工具
- LLM 自行判断是否继续
- 每隔 `soft_limit_interval` 步重复提醒

#### 硬限制模式（适合短任务 / 子任务）

- 达到 `max_steps` 后，在下次 LLM 调用时不传 tools 参数
- 注入 `hard_limit_prompt`（assistant prefill）
- LLM 只能输出文本总结
- 该轮结束后直接终止循环

#### Doom Loop 检测

参照 opencode `processor.ts:152-177`：
- 记录最近 N 次工具调用的 `(tool_name, arguments_json)` 签名
- 连续 `doom_loop_threshold` 次完全相同 → 产出 ERROR 事件 → 终止循环

#### 去掉超时限制

opencode 不使用超时机制。步数限制 + token 限制已经足够。
如果用户需要超时控制，可在 Agent 层面用 `asyncio.timeout()` 自行实现。

#### LimitChecker 保留三态

```python
class LimitChecker:
    def check(self) -> LimitResult:
        # "continue" - 继续
        # "warn" - 软限制提醒（仅 soft 模式）
        # "stop" - 停止（硬限制到达 / token 耗尽 / doom loop）
```

`is_last_step` 作为独立属性，供循环体判断是否禁用工具。

#### 去掉的内容

- `timeout` 字段
- `timeout_prompt` 字段
- `_timeout_checkpoint` 机制
- 周期性步数检查点的"checkpoint"概念（简化为单一阈值 + 可选间隔）

## 模块 4：Skill 引导重构

### 集成在 `prompts.py` 的第 4 段

Skill 系统代码（`skill/` 目录）完全不动，仅改系统提示词中的引导文本。

详见模块 1 的第 4 段描述。

## 影响范围

### 需要修改的文件

| 文件 | 改动类型 |
|------|---------|
| `prompts.py` | 完全重写内容 |
| `builtin_tools.py` | 扩展四状态 + priority + 重写工具描述 |
| `loop.py` | 双模式限制 + doom loop + 去超时 |
| `limits.py` | 重构 LoopLimits + LimitChecker |
| `agent.py` | 适配新的 LoopLimits 接口 |
| `events.py` | 无改动（SOFT_LIMIT 保留） |

### 需要更新的测试

| 测试文件 | 原因 |
|---------|------|
| `test_prompts.py` | 提示词内容变了 |
| `test_builtin_tools.py` | 新增状态和 priority |
| `test_loop.py` | 双模式限制 + doom loop |
| `test_limits.py` | 新的 LoopLimits 接口 |
| `test_agent.py` | 可能需要适配 |

### 不改动的文件

- `tool.py` — 工具系统
- `events.py` — 事件系统
- `retry.py` — 重试机制
- `renderer.py` — 渲染器
- `llm/` — LLM 抽象层
- `skill/` — Skill 系统代码
- `errors.py` — 异常体系
