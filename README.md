# pure-agent-loop

轻量级 ReAct 模式 Agentic Loop 框架。

## 安装

```bash
pip install pure-agent-loop
```

## 快速开始

```python
from pure_agent_loop import Agent, tool

@tool
def search(query: str) -> str:
    """搜索网页"""
    return f"搜索结果: {query}"

agent = Agent(model="gpt-4o-mini", tools=[search])
result = agent.run("帮我搜索 Python 教程")
print(result.content)
```

## 许可证

MIT
