"""
MCP Server - 知识库检索工具

通过 stdio 协议暴露知识库检索能力，供支持 MCP 的宿主（如 Claude Desktop）调用。
与 qa_agent 共享同一套工具定义（qa_tools.py），保证本地直调和 MCP 远程调用的行为一致。

工具结果的约定：
    返回 JSON 信封 {"text": ..., "hits": [...]}。
        text -> 给模型看的摘要文本
        hits -> 结构化命中片段（供内部 agent 走 MCP 时重建证据；外部宿主忽略即可）

启动：
    cd backend
    python -m app.mcp_server

    或在 Claude Desktop 配置 mcpServers：
    {
      "knowledge-manager": {
        "command": "python",
        "args": ["-m", "app.mcp_server"],
        "env": { "PYTHONPATH": "F:/agent/agentuse/backend" }
      }
    }
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 确保项目根目录在路径里，即使从别的目录启动也能找到 app.config
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from app.agents.qa_tools import QAToolbox, TOOL_SCHEMAS  # noqa: E402
from app.config import get_settings  # noqa: E402

settings = get_settings()

# MCP 用 stdout 传 JSON-RPC 协议，stderr 可能被宿主管道化。loguru 默认写 stderr，
# 一旦 search_kb 加载 BGE 模型时打出大量日志，会塞满 stderr 管道导致死锁。
# 所以把服务端日志重定向到文件，保证 stdio 通道干净。
from loguru import logger  # noqa: E402

_log_dir = backend_root / "data" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(str(_log_dir / "mcp_server.log"), rotation="10 MB", retention=3)


def _convert_schema(fn_schema: Dict[str, Any]) -> Tool:
    """把 OpenAI 风格的 function schema 转成 MCP Tool"""
    fn = fn_schema["function"]
    return Tool(
        name=fn["name"],
        description=fn["description"],
        inputSchema=fn.get("parameters") or {"type": "object", "properties": {}},
    )


MCP_TOOLS = [_convert_schema(s) for s in TOOL_SCHEMAS]

app = Server("knowledge-manager")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用工具"""
    return MCP_TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> list[TextContent]:
    """执行工具调用，返回 JSON 信封 {text, hits}"""
    toolbox = QAToolbox(
        top_k=settings.search_top_k,
        document_ids=None,
        use_name_match=True,
    )
    text = toolbox.call(name, arguments or {})
    envelope = {"text": text, "hits": toolbox.collected()}
    return [TextContent(type="text", text=json.dumps(envelope, ensure_ascii=False))]


async def main() -> None:
    """启动 stdio MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
