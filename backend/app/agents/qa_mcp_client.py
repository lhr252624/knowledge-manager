"""
QA MCP Client - 通过 MCP 协议调用工具的代理层

当 settings.qa_tool_transport = "mcp" 时，qa_agent 用这个类代替 QAToolbox。
接口与 QAToolbox 一致：call() 返回给模型看的文本，同时把结构化命中收进 evidence。

为什么用"后台事件循环 + 常驻会话"而不是每次调用都起一个新子进程：
    search_kb 会加载 BGE 模型（约 10s），如果每次调用都重起 server，agent 循环里
    3~6 次工具调用就要重新加载 3~6 遍模型。所以这里在后台线程里起一个常驻的
    asyncio 事件循环，维持同一个 MCP 会话，BGE 只加载一次。

证据怎么来：
    MCP 协议本身只传文本，但 agent 最终作答需要的是"命中的片段"而不是字符串。
    所以 mcp_server 的工具结果约定为 JSON 信封 {"text": ..., "hits": [...]}：
        text -> 给模型看的摘要（与 local 模式一致）
        hits -> 结构化片段（document_id / chunk_index / content / title ...）
    客户端解析后把 hits 收进 evidence，text 原样交给模型。

使用前：
    pip install "mcp>=1.3.0,<2.0.0"
    （无需手动启动 server，客户端按需拉起 stdio 子进程）
"""
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.agents.qa_tools import MAX_TOTAL_CHARS


class QAMCPToolbox:
    """通过 MCP 调用工具的代理，接口与 QAToolbox 一致"""

    def __init__(
        self,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        use_name_match: bool = True,
    ) -> None:
        self.top_k = top_k
        self.document_ids = document_ids
        self.use_name_match = use_name_match
        self.evidence: Dict[tuple, Dict[str, Any]] = {}
        self.chars = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._stdio_ctx = None

    # ---------- 会话生命周期 ----------
    def _start(self) -> None:
        """惰性启动后台事件循环并建立常驻 MCP 会话（只做一次）"""
        if self._loop is not None:
            return
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="qa-mcp-client"
            )
            self._thread.start()
            self._session = asyncio.run_coroutine_threadsafe(
                self._open_session(), self._loop
            ).result(timeout=60)
            logger.info("MCP 常驻会话已建立")
        except Exception as exc:
            logger.error(f"MCP 会话建立失败: {exc}")
            raise RuntimeError(
                "MCP transport 不可用：请确认已 pip install 'mcp>=1.3.0,<2.0.0'，"
                "或把 QA_TOOL_TRANSPORT 改回 local"
            ) from exc

    async def _open_session(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        backend_root = Path(__file__).resolve().parents[2]
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            env={**os.environ, "PYTHONPATH": str(backend_root)},
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        return session

    def close(self) -> None:
        """关闭会话并停掉后台循环（无会话时是 no-op）"""
        if self._loop is None:
            return

        async def _shutdown():
            try:
                if self._session is not None:
                    await self._session.__aexit__(None, None, None)
            except Exception:  # 会话关闭失败不影响收尾
                pass
            try:
                if self._stdio_ctx is not None:
                    await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=10)
        except Exception as exc:
            logger.warning(f"MCP 会话关闭异常: {exc}")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
            self._session = None
            self._stdio_ctx = None

    # ---------- 执行 ----------
    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> str:
        self._start()
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, arguments=args or {}), self._loop
            )
            result = fut.result(timeout=60)
        except Exception as exc:
            logger.warning(f"MCP 工具 {name} 调用失败: {exc}")
            return f"工具执行失败：{exc}"

        if not result.content or not result.content[0].text:
            return ""
        raw = result.content[0].text
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw  # 非 JSON，原样返回（如错误提示）

        text = envelope.get("text") or ""
        for hit in envelope.get("hits") or []:
            self.add(hit)
        return text

    # ---------- 证据 ----------
    def add(self, chunk: Dict[str, Any]) -> bool:
        """收进证据；已存在或超预算则返回 False"""
        key = (int(chunk["document_id"]), int(chunk.get("chunk_index", 0)))
        if key in self.evidence:
            return False
        if self.chars >= MAX_TOTAL_CHARS:
            return False
        self.evidence[key] = chunk
        self.chars += len(chunk.get("content") or "")
        return True

    def collected(self) -> List[Dict[str, Any]]:
        return list(self.evidence.values())

    def is_empty(self) -> bool:
        return not self.evidence
