"""Exa 搜索示例: 使用 Exa AI 进行网页搜索 + 网页内容抓取（流式事件输出 + 思考模式）

功能:
    - exa_search: 使用 Exa AI 搜索网页，返回标题、URL 和摘要
    - fetch_webpage: 访问指定 URL，将 HTML 转换为 Markdown 格式返回

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv requests rich html2text
    cp .env.example .env
    # 编辑 .env 填入 API_KEY 和 EXA_API_KEY

支持的思考模式（通过 THINKING_LEVEL 环境变量配置）:
    off    - 关闭思考模式（默认）
    low    - 低深度思考
    medium - 中等深度思考
    high   - 高深度思考

注意: 思考模式需要支持推理的模型（如 DeepSeek-R1, o1, o3 等）
"""

import asyncio
import os
import re
from pathlib import Path

import html2text
import requests
from dotenv import load_dotenv

from pure_agent_loop import Agent, tool, ThinkingLevel
from rich_renderer import RichRenderer

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")

# Exa API 配置
EXA_API_URL = "https://api.exa.ai/search"
EXA_API_KEY = os.getenv("EXA_API_KEY", "")


@tool
def exa_search(query: str, num_results: int = 5) -> str:
    """使用 Exa AI 搜索网页内容

    Args:
        query: 搜索查询词
        num_results: 返回结果数量，默认5条
    """
    if not EXA_API_KEY:
        return "错误: 未配置 EXA_API_KEY 环境变量"

    try:
        response = requests.post(
            EXA_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": EXA_API_KEY,
            },
            json={
                "query": query,
                "type": "auto",
                "numResults": num_results,
                "contents": {"text": True},
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # 格式化搜索结果
        results = data.get("results", [])
        if not results:
            return f"未找到关于 '{query}' 的搜索结果"

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            # 截取文本内容前500字符
            text = result.get("text", "")[:500]
            if len(result.get("text", "")) > 500:
                text += "..."

            formatted.append(f"[{i}] {title}\n    URL: {url}\n    内容: {text}\n")

        return "\n".join(formatted)

    except requests.exceptions.Timeout:
        return "错误: 搜索请求超时"
    except requests.exceptions.RequestException as e:
        return f"错误: 搜索请求失败 - {e}"
    except Exception as e:
        return f"错误: {e}"


@tool
def fetch_webpage(url: str, max_length: int = 8000) -> str:
    """访问指定 URL 并将网页内容转换为 Markdown 格式返回

    适用于需要获取网页详细内容的场景，会自动剔除 HTML 样式和脚本，
    保留文本结构（标题、段落、链接、列表等）。

    Args:
        url: 要访问的网页 URL（必须是完整的 http/https 地址）
        max_length: 返回内容的最大字符数，默认 8000
    """
    if not url.startswith(("http://", "https://")):
        return "错误: URL 必须以 http:// 或 https:// 开头"

    try:
        # 设置请求头模拟浏览器访问
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # 尝试检测编码
        response.encoding = response.apparent_encoding or "utf-8"
        html_content = response.text

        # 配置 html2text 转换器
        converter = html2text.HTML2Text()
        converter.ignore_links = False  # 保留链接
        converter.ignore_images = True  # 忽略图片
        converter.ignore_emphasis = False  # 保留强调
        converter.skip_internal_links = True  # 跳过页内锚点
        converter.inline_links = True  # 使用行内链接格式
        converter.protect_links = True  # 保护链接不被换行
        converter.body_width = 0  # 不自动换行
        converter.unicode_snob = True  # 使用 Unicode 字符
        converter.ignore_tables = False  # 保留表格

        # 转换为 Markdown
        markdown_content = converter.handle(html_content)

        # 清理多余空行（连续3个以上空行压缩为2个）
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        markdown_content = markdown_content.strip()

        # 截断过长内容
        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "\n\n... [内容已截断]"

        if not markdown_content:
            return f"警告: 网页 {url} 未能提取到有效文本内容"

        return f"# 网页内容 ({url})\n\n{markdown_content}"

    except requests.exceptions.Timeout:
        return f"错误: 访问 {url} 超时"
    except requests.exceptions.TooManyRedirects:
        return f"错误: {url} 重定向次数过多"
    except requests.exceptions.HTTPError as e:
        return f"错误: HTTP 请求失败 - {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"错误: 网络请求失败 - {e}"
    except Exception as e:
        return f"错误: 处理网页内容时出错 - {e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        # 安全地计算数学表达式
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含不允许的字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


async def main():
    # 检查必要的配置
    if not os.getenv("API_KEY"):
        print("错误: 请在 .env 文件中配置 API_KEY")
        return
    if not EXA_API_KEY:
        print("错误: 请在 .env 文件中配置 EXA_API_KEY")
        return

    # 获取思考模式配置
    thinking_level: ThinkingLevel = os.getenv("THINKING_LEVEL", "off")  # type: ignore
    emit_reasoning = thinking_level != "off"

    if emit_reasoning:
        print(f"🧠 思考模式已启用: {thinking_level}")

    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[exa_search, fetch_webpage, calculate],
        system_prompt=(
            "你是一个专业的搜索助手，擅长从互联网获取信息。\n\n"
            "工具使用策略：\n"
            "1. exa_search: 用于搜索相关网页，获取标题、URL 和简短摘要\n"
            "2. fetch_webpage: 当需要查看某个网页的详细内容时使用，传入 URL 获取完整的 Markdown 格式内容\n"
            "3. calculate: 用于数学计算\n\n"
            "工作流程建议：\n"
            "- 先用 exa_search 搜索相关信息\n"
            "- 如果搜索结果的摘要不够详细，使用 fetch_webpage 获取感兴趣的网页完整内容\n"
            "- 综合所有信息回答用户问题，并注明信息来源"
        ),
        thinking_level=thinking_level,
        emit_reasoning_events=emit_reasoning,
    )

    # 测试查询
    query = "我想了解一下灵迹岛这款桌游的规则"
    print(f"\n🔍 查询: {query}\n")
    print("=" * 60)

    # 使用 RichRenderer 美化输出
    renderer = RichRenderer(
        max_thought_lines=3,      # 思考内容最多显示3行
        max_result_chars=150,     # 工具结果最多显示150字符
        show_todo_table=True,     # 用表格显示 Todo 列表
        max_reasoning_lines=15,   # 推理内容最多显示15行
    )

    # 流式执行，实时输出事件
    async for event in agent.arun_stream(query):
        renderer.render(event)

    print("\n" + "=" * 60)
    print("✅ 执行完成")


if __name__ == "__main__":
    asyncio.run(main())
