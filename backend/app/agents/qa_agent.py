"""
QAAgent - 检索增强问答智能体

流程（RAG）：
    问题 -> 向量召回 Top-K 片段 -> 构造上下文 -> LangChain Chain -> 答案 + 引用
"""
from typing import Any, AsyncIterator, Dict, List, Optional


from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.services.milvus_store import milvus_store

settings = get_settings()

_SYSTEM_PROMPT = """你是「智能知识管理系统」的问答助手。请严格依据用户提供的【参考资料】回答问题。

回答要求：
1. 只使用参考资料中的信息，不要编造。资料不足时明确说明"资料中未提及"。
2. 用中文回答，条理清晰；涉及步骤或要点时使用列表。
3. 在引用具体内容处用 [1] [2] 这样的编号标注来源，编号对应参考资料序号。
4. 参考资料标题里标注了文档名（文件名）与标题。如果用户直接提到某个文档名，
   请优先依据那份文档作答，并在回答开头点明是哪份文档。
5. 如果提供了【对话历史】，结合上下文理解追问（例如"那第二点呢"），
   但事实依据仍然只能来自参考资料。
6. 不要重复罗列资料原文，要归纳总结。
"""

_USER_TEMPLATE = "{prompt}"


class QAAgent(BaseAgent):
    """知识库问答智能体"""

    name = "qa_agent"
    description = "基于知识库检索结果回答问题，并给出引用来源"

    # ---------- 提示词构建 ----------
    @staticmethod
    def _build_prompt(
        context: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """组装 human 消息：对话历史 + 参考资料 + 问题"""
        blocks: List[str] = []
        if history:
            lines = []
            for item in history:
                who = "用户" if item.get("role") == "user" else "助手"
                lines.append(f"{who}：{item.get('content', '')}")
            blocks.append("【对话历史】\n" + "\n".join(lines))
        blocks.append("【参考资料】\n" + context)
        blocks.append("【问题】\n" + question)
        blocks.append("请基于参考资料作答。")
        return "\n\n".join(blocks)

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "（无相关资料）"
        blocks = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("title") or "未知文档"
            filename = c.get("filename")
            # 文件名单独标出来：很多文档的标题是从正文首行抽的，
            # 和用户口中的"那个文件"对不上，把原名列出来模型才认得出
            source = f"《{title}》"
            if filename and filename != title:
                source += f"（文件名：{filename}）"
            blocks.append(
                f"[{i}] 来源：{source} 第{c.get('chunk_index', 0) + 1}片段\n"
                f"{c.get('content', '').strip()}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        cites = []
        for i, c in enumerate(chunks, start=1):
            key = (c["document_id"], c["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            cites.append(
                {
                    "index": i,
                    "document_id": c["document_id"],
                    "title": c.get("title") or c.get("filename"),
                    "filename": c.get("filename"),
                    "chunk_index": c.get("chunk_index"),
                    "score": c.get("score"),
                    "vector_score": c.get("vector_score"),
                    "name_score": c.get("name_score", 0.0),
                    "match_type": c.get("match_type", "vector"),
                    "preview": (c.get("content") or "")[:200],
                }
            )
        return cites

    # ---------- 检索 ----------
    def _retrieve(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[List[int]],
        use_name_match: bool = True,
    ) -> List[Dict[str, Any]]:
        from app.services.retrieval_service import retrieve

        hits = retrieve(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
            use_name_match=use_name_match,
        )["hits"]
        if not hits:
            return hits

        # 补充标题（文件名优先用 Milvus 里记录的原值）
        from app.db.database import session_scope
        from app.models.document import Document

        ids = {h["document_id"] for h in hits}
        with session_scope() as db:
            rows = db.query(Document).filter(Document.id.in_(ids)).all()
            meta_map = {
                r.id: {"title": r.title or r.filename, "filename": r.filename}
                for r in rows
            }

        for h in hits:
            meta = meta_map.get(h["document_id"], {})
            h["title"] = meta.get("title") or h.get("filename")
            h["filename"] = meta.get("filename") or h.get("filename")
        return hits

    # ---------- 同步问答 ----------
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            question: 用户问题（必填）
            top_k: 召回条数
            document_ids: 限定文档范围
            history: 最近几轮对话 [{"role": "user"/"assistant", "content": str}]
        """
        question = (kwargs.get("question") or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        document_ids = kwargs.get("document_ids")
        use_name_match = kwargs.get("use_name_match", True)
        history = kwargs.get("history") or []

        chunks = self._retrieve(question, top_k, document_ids, use_name_match)
        if not chunks:
            return {
                "question": question,
                "answer": "知识库中暂未检索到与该问题相关的内容。请先上传相关文档，或换一种问法试试。",
                "citations": [],
                "retrieved": 0,
            }

        context = self._build_context(chunks)
        answer = self._run_chain(question, context, history)

        return {
            "question": question,
            "answer": answer,
            "citations": self._citations(chunks),
            "retrieved": len(chunks),
        }

    def _run_chain(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """LangChain LCEL 链：Prompt | LLM | StrOutputParser"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_TEMPLATE),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"prompt": self._build_prompt(context, question, history)}
        ).strip()

    # ---------- 流式问答 ----------
    async def astream(self, **kwargs: Any) -> AsyncIterator[str]:
        """SSE 流式输出（逐段 yield 文本）"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        question = (kwargs.get("question") or "").strip()
        if not question:
            yield "问题不能为空"
            return

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        document_ids = kwargs.get("document_ids")
        use_name_match = kwargs.get("use_name_match", True)
        history = kwargs.get("history") or []
        # 流式接口会先调 prepare_stream 拿引用，把检索结果传进来避免重复检索
        # （重复一次 = 多一次 BGE 编码 + 多一次 Milvus 查询）
        chunks = kwargs.get("chunks")
        if chunks is None:
            chunks = self._retrieve(question, top_k, document_ids, use_name_match)

        if not chunks:
            yield "知识库中暂未检索到与该问题相关的内容。请先上传相关文档，或换一种问法试试。"
            return

        context = self._build_context(chunks)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()

        payload = {"prompt": self._build_prompt(context, question, history)}
        async for piece in chain.astream(payload):
            if piece:
                yield piece

    def prepare_stream(self, **kwargs: Any) -> Dict[str, Any]:
        """SSE 场景下先返回引用信息，再流式吐答案"""
        question = (kwargs.get("question") or "").strip()
        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        chunks = self._retrieve(
            question,
            top_k,
            kwargs.get("document_ids"),
            kwargs.get("use_name_match", True),
        )
        return {
            "chunks": chunks,
            "citations": self._citations(chunks) if chunks else [],
        }

    def health(self) -> Dict[str, Any]:
        ok = milvus_store.health()
        return {"agent": self.name, "milvus": ok, "status": "ok" if ok else "down"}


qa_agent = QAAgent()
