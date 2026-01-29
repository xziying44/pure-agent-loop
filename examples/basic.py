"""基础示例: 最简 Agent 使用"""

from pure_agent_loop import Agent, tool


@tool
def search(query: str) -> str:
    """搜索网页内容

    Args:
        query: 搜索关键词
    """
    # 这里替换为实际的搜索实现
    return f"搜索 '{query}' 的结果: Python 是一种通用编程语言..."


@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search, calculate],
        system_prompt="你是一个有用的助手，可以搜索信息和计算数学表达式。",
    )

    result = agent.run("Python 语言是什么时候发布的？1991 年到 2026 年一共多少年？")
    print(f"回答: {result.content}")
    print(f"步数: {result.steps}")
    print(f"终止原因: {result.stop_reason}")


if __name__ == "__main__":
    main()
