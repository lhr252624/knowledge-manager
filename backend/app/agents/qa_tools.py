"""
QA agent 的工具集

这里只定义"能做什么"，不定义"什么时候做"——后者交给模型（见 qa_agent 的循环）。
同一份工具注册表对外有两个入口：
    - 本地直调（默认，无进程间开销）
    - MCP server（backend/app/mcp_server.py，可被 Claude Desktop 等宿主使用）

为什么工具要自己攒证据，而不是各自返回字符串就完事：
    模型在不同步骤里可能用不同措辞反复检索，答案要基于"所有看到过的片段"，
    而不是最后一次检索的结果。所以工具把命中的片段按 (文档, 片段) 去重后
    收进 evidence，最后由作答环节统一使用。
"""
from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import get_settings

settings = get_settings()

# 单次问答允许读取的总字数上限，防止模型把整库拖进上下文
MAX_TOTAL_CHARS = 16000
# 单次 read_document 最多读几块
MAX_READ_CHUNKS = 8
# 单个片段喂给模型时截断到多少字（完整内容仍在 evidence 里）
SNIPPET_CHARS = 300

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "在知识库里做语义检索，返回最相关的片段。可以换不同措辞多次调用。"
                "第一次结果不理想时，应当改写关键词再搜，而不是直接放弃。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句或关键词"},
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 5，最多 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_filename",
            "description": "按文件名/标题查找文档，返回文档 id 与分块数；适合用户点名某份文档时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "文件名或标题的一部分"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "按分块区间读取某篇文档的原文，用于通读全文或查看检索没覆盖到的部分。"
                f"单次最多读 {MAX_READ_CHUNKS} 块，请先用 search_by_filename 拿到文档 id 和分块数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "文档 id"},
                    "start_chunk": {
                        "type": "integer",
                        "description": "起始分块下标，从 0 开始",
                    },
                    "end_chunk": {"type": "integer", "description": "结束分块下标（含）"},
                },
                "required": ["document_id", "start_chunk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出知识库里所有可检索的文档（id、标题、分块数）。用户问'都有哪些文档'或需要跨文档对比时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


class QAToolbox:
    """一次问答会话的工具箱：执行工具、累积证据、记录轨迹"""

    def __init__(
        self,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        use_name_match: bool = True,
    ) -> None:
        self.top_k = top_k
        self.document_ids = document_ids
        self.use_name_match = use_name_match
        # (document_id, chunk_index) -> 片段，保持首次出现的顺序
        self.evidence: Dict[tuple, Dict[str, Any]] = {}
        self.chars = 0

    # ---------- 执行 ----------
    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> str:
        args = args or {}
        try:
            if name == "search_kb":
                return self._search_kb(args)
            if name == "search_by_filename":
                return self._search_by_filename(args)
            if name == "read_document":
                return self._read_document(args)
            if name == "list_documents":
                return self._list_documents()
            return f"未知工具：{name}。可用工具：{', '.join(TOOL_NAMES)}"
        except Exception as exc:  # 工具出错交回模型自己处理，不中断循环
            logger.warning(f"QA 工具 {name} 执行失败: {exc}")
            return f"工具执行失败：{exc}"

    # ---------- 证据 ----------
    def add(self, chunk: Dict[str, Any]) -> bool:
        """收进证据；已存在则返回 False"""
        key = (int(chunk["document_id"]), int(chunk.get("chunk_index", 0)))
        if key in self.evidence:
            return False
        self.evidence[key] = chunk
        self.chars += len(chunk.get("content") or "")
        return True

    def collected(self) -> List[Dict[str, Any]]:
        return list(self.evidence.values())

    def is_empty(self) -> bool:
        return not self.evidence

    def close(self) -> None:
        """本地直调无需资源释放；与 QAMCPToolbox 保持接口一致（no-op）"""

    # ---------- 具体工具 ----------
    def _search_kb(self, args: Dict[str, Any]) -> str:
        from app.services.retrieval_service import retrieve

        query = str(args.get("query") or "").strip()
        if not query:
            return "query 不能为空"
        try:
            top_k = int(args.get("top_k") or self.top_k)
        except (TypeError, ValueError):
            top_k = self.top_k
        top_k = max(1, min(top_k, 10))

        data = retrieve(
            query=query,
            top_k=top_k,
            document_ids=self.document_ids,
            use_name_match=self.use_name_match,
        )
        hits = data.get("hits") or []
        if not hits:
            return f"没有检索到与「{query}」相关的片段。可以换一组关键词再试。"

        lines = [f"检索「{query}」命中 {len(hits)} 条："]
        for i, hit in enumerate(hits, start=1):
            added = self.add(hit)
            title = hit.get("title") or hit.get("filename") or "未知文档"
            flag = "" if added else "（已读过）"
            snippet = (hit.get("content") or "").strip().replace("\n", " ")[:SNIPPET_CHARS]
            lines.append(
                f"[{i}] 《{title}》 第{int(hit.get('chunk_index', 0)) + 1}片段 "
                f"分数{float(hit.get('score') or 0):.3f}{flag}\n{snippet}"
            )
        lines.append(f"（当前已累积 {len(self.evidence)} 个片段，{self.chars} 字）")
        return "\n".join(lines)

    def _search_by_filename(self, args: Dict[str, Any]) -> str:
        from app.services.retrieval_service import name_matches

        name = str(args.get("name") or "").strip()
        if not name:
            return "name 不能为空"

        matched = name_matches(name, self.document_ids)
        if not matched:
            return f"没有找到名字匹配「{name}」的文档。可以用 list_documents 看看都有哪些。"

        from app.db.database import session_scope
        from app.models.document import Document

        lines = [f"名字匹配「{name}」的文档："]
        with session_scope() as db:
            for doc_id, info in list(matched.items())[:5]:
                doc = db.get(Document, doc_id)
                if doc is None:
                    continue
                lines.append(
                    f"- id={doc_id}  {info['text']}  分块数={doc.chunk_count or 0}  "
                    f"（匹配度 {info['score']}）"
                )
        lines.append("需要看原文请用 read_document(document_id, start_chunk, end_chunk)。")
        return "\n".join(lines)

    def _read_document(self, args: Dict[str, Any]) -> str:
        from app.db.database import session_scope
        from app.models.document import Chunk, Document

        try:
            doc_id = int(args.get("document_id"))
            start = max(0, int(args.get("start_chunk") or 0))
        except (TypeError, ValueError):
            return "document_id / start_chunk 必须是整数"
        try:
            end = int(args.get("end_chunk", start + MAX_READ_CHUNKS - 1))
        except (TypeError, ValueError):
            end = start + MAX_READ_CHUNKS - 1
        if end < start:
            start, end = end, start
        end = min(end, start + MAX_READ_CHUNKS - 1)

        with session_scope() as db:
            doc = db.get(Document, doc_id)
            if doc is None:
                return f"文档 {doc_id} 不存在"
            rows = (
                db.query(Chunk)
                .filter(Chunk.document_id == doc_id)
                .order_by(Chunk.chunk_index)
                .offset(start)
                .limit(end - start + 1)
                .all()
            )
            total = doc.chunk_count or 0
            title = doc.title or doc.filename

        if not rows:
            return f"文档 {doc_id} 在分块 {start}~{end} 没有内容（共 {total} 块）"

        if self.chars >= MAX_TOTAL_CHARS:
            return f"本次问答的读取预算已用尽（{self.chars} 字），请基于已有内容作答。"

        blocks = [f"《{title}》第 {start + 1}~{start + len(rows)} 块（共 {total} 块）："]
        for r in rows:
            self.add(
                {
                    "document_id": doc_id,
                    "chunk_index": r.chunk_index,
                    "content": r.content,
                    "filename": doc.filename,
                    "title": title,
                    "score": 0.0,
                    "match_type": "read",
                }
            )
            blocks.append(f"【第{r.chunk_index + 1}块】{r.content}")
        blocks.append(f"（当前已累积 {len(self.evidence)} 个片段，{self.chars} 字）")
        return "\n".join(blocks)

    def _list_documents(self) -> str:
        from app.db.database import session_scope
        from app.models.document import Document, DocumentStatus

        with session_scope() as db:
            rows = (
                db.query(Document)
                .filter(Document.status == DocumentStatus.READY)
                .order_by(Document.id.desc())
                .limit(30)
                .all()
            )
            items = [
                (r.id, r.title or r.filename, r.filename, r.chunk_count or 0) for r in rows
            ]

        if not items:
            return "知识库里还没有可检索的文档。"

        lines = [f"知识库共 {len(items)} 篇文档（最多列出 30 篇）："]
        for doc_id, title, filename, chunks in items:
            lines.append(f"- id={doc_id}  {title}  分块={chunks}")
        return "\n".join(lines)
