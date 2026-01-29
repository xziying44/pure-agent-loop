"""自定义渲染器示例: 定制事件展示格式"""

from pure_agent_loop import Agent, tool, Renderer, Event, EventType


@tool
def search(query: str) -> str:
    """搜索网页

    Args:
        query: 搜索关键词
    """
    return f"找到 10 条关于 '{query}' 的结果"


@tool
def save_file(filename: str, content: str) -> str:
    """保存文件

    Args:
        filename: 文件名
        content: 文件内容
    """
    return f"已保存到 {filename}"


# 创建自定义渲染器
renderer = Renderer()


@renderer.on_tool("search")
def render_search(event: Event) -> str:
    """自定义搜索工具的渲染"""
    if event.type == EventType.ACTION:
        return f"🔍 正在搜索: {event.data['args']['query']}"
    return f"📋 搜索完成: {event.data.get('result', '')}"


@renderer.on_tool("save_file")
def render_save(event: Event) -> str:
    """自定义保存文件工具的渲染"""
    if event.type == EventType.ACTION:
        return f"💾 正在保存: {event.data['args']['filename']}"
    return f"✅ 保存完成"


@renderer.on_event(EventType.SOFT_LIMIT)
def render_limit(event: Event) -> str:
    return f"⏰ 提醒: AI 正在调整策略 ({event.data['reason']})"


def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search, save_file],
    )

    for event in agent.run_stream("搜索 Python 教程并保存到文件"):
        output = renderer.render(event)
        if output:
            print(output)


if __name__ == "__main__":
    main()
