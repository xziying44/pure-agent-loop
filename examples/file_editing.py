"""文件编辑示例: 使用 Sandbox + 文件工具让 Agent 编写登录注册页面

演示 pure-agent-loop 的沙箱系统和文件工具能力。
Agent 将使用 file_write 工具自动生成完整的 HTML/CSS/JS 登录注册页面。

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv
    cp .env.example .env
    # 编辑 .env 填入实际的 API 密钥

运行:
    cd examples
    python file_editing.py

运行后会在 examples/output/login-page/ 目录下生成:
    ├── index.html
    ├── style.css
    └── script.js
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pure_agent_loop import Agent, Sandbox, Renderer, LoopLimits

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "output" / "login-page"


async def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 配置沙箱：Agent 只能在 output 目录内读写文件
    sandbox = Sandbox(
        write_paths=[str(OUTPUT_DIR)],
        read_paths=[str(OUTPUT_DIR)],
    )

    agent = Agent(
        name="前端开发助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        sandbox=sandbox,
        # 分步编码 + 自查验证需要较多轮次
        limits=LoopLimits(max_steps=20),
        system_prompt=(
            "你是一个严谨的前端开发工程师，擅长编写高质量的 Web 页面。\n\n"
            "## 工作流程\n"
            "收到用户需求后，严格按以下步骤执行：\n"
            "1. **需求分析** — 理解用户意图，明确功能点和交付物\n"
            "2. **制定计划** — 列出文件清单、各文件职责和实现顺序\n"
            "3. **逐步编码** — 先搭建 HTML 骨架，再编写 CSS 样式，最后实现 JS 交互；"
            "每个文件单独创建，不要一次性输出所有代码\n"
            "4. **自查验证** — 编码完成后，逐一读取所有已创建的文件，检查：\n"
            "   - 文件间引用路径是否正确\n"
            "   - HTML 结构与 JS 选择器是否匹配\n"
            "   - CSS 类名与 HTML 是否一致\n"
            "   - 是否存在语法错误或遗漏\n"
            "5. **修复问题** — 如果自查发现问题，立即修复\n"
            "6. **汇报结果** — 总结已完成的工作、文件清单和使用方式\n\n"
            "## 编码规范\n"
            "- 文件之间通过相对路径引用\n"
            "- HTML 语义化标签，CSS 不使用 !important\n"
            "- JS 使用原生 DOM API，无外部依赖\n"
            "- 代码注释简洁清晰"
        ),
    )

    task = f"""请在 {OUTPUT_DIR} 目录设计一个大模型聊天web，要求前端用户可以登录注册，实现简单的聊天。后台管理员固定账号密码为admin，可以查看注册的用户，和查看用户的聊天记录。"""

    renderer = Renderer()

    print("🚀 Agent 开始生成登录注册页面...\n")

    async for event in agent.arun_stream(task):
        output = renderer.render(event)
        if output:
            print(output)


    task = f"""请在 {OUTPUT_DIR} 的基础上新增一个日志管理功能。"""

    renderer = Renderer()

    print("🚀 Agent 开始生成登录注册页面...\n")

    async for event in agent.arun_stream(task):
        output = renderer.render(event)
        if output:
            print(output)


if __name__ == "__main__":
    asyncio.run(main())
