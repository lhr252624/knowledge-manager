"""
SummaryAgent - 摘要与关键词智能体
"""
import json
import re
from typing import Any, Dict

from loguru import logger

from app.agents.base_agent import BaseAgent
from app.db.database import session_scope
from app.models.document import Document

_SUMMARY_SYSTEM = (
    "你是一名专业的中文文档分析助手。请严格按用户要求输出，不要添加多余的解释。"
)

_JSON_RE = re.compile(r"\{.*\}", re.S)


class SummaryAgent(BaseAgent):
    """文档摘要智能体：生成摘要 + 关键词"""

    name = "summary_agent"
    description = "为文档生成中文摘要与关键词"

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            text: 正文（与 document_id 二选一）
            document_id: 从数据库读取文档分块
            filename: 文件名，用于提示
            persist: 是否写回 MySQL，默认 False
        """
        text = kwargs.get("text")
        document_id = kwargs.get("document_id")
        filename = kwargs.get("filename", "")
        persist = kwargs.get("persist", False)

        if not text and document_id:
            text = self._load_text(document_id)

        if not text:
            raise ValueError("缺少可用文本")

        # 控制输入长度：摘要只需把握主题与结论，喂全文反而更慢且容易跑题
        excerpt = text[:2500]

        prompt = (
            f"文档名称：{filename}\n"
            "请阅读以下文档内容，输出一个 JSON 对象，不要输出任何其他内容：\n"
            "{\n"
            '  "summary": "150字以内的中文摘要，说明文档主题与核心结论",\n'
            '  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]\n'
            "}\n\n"
            f"文档内容：\n{excerpt}"
        )

        # 摘要输出很短（摘要+5个关键词 ≈ 200 tokens），限制上限可明显缩短耗时
        raw = self.invoke_llm(prompt, system=_SUMMARY_SYSTEM, max_tokens=400)

        summary, keywords = self._parse(raw)

        if persist and document_id:
            with session_scope() as db:
                doc = db.get(Document, document_id)
                if doc is not None:
                    doc.summary = summary
                    doc.keywords = json.dumps(keywords, ensure_ascii=False)
            logger.info(f"文档 {document_id} 摘要已写入")

        return {"summary": summary, "keywords": keywords}

    # ---------- 内部方法 ----------
    def _load_text(self, document_id: int) -> str:
        from app.models.document import Chunk

        with session_scope() as db:
            rows = (
                db.query(Chunk)
                .filter(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
                .limit(20)
                .all()
            )
        return "\n".join(r.content for r in rows)

    @staticmethod
    def _parse(raw: str) -> tuple[str, list]:
        """容错解析 LLM 返回的 JSON"""
        match = _JSON_RE.search(raw or "")
        if match:
            try:
                data = json.loads(match.group(0))
                summary = str(data.get("summary", "")).strip()
                keywords = data.get("keywords") or []
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in re.split(r"[,，;；]", keywords) if k.strip()]
                return summary, [str(k) for k in keywords][:12]
            except json.JSONDecodeError:
                pass
        # 解析失败则退化为纯文本
        return (raw or "").strip()[:600], []

    def health(self) -> Dict[str, Any]:
        return {"agent": self.name, "status": "ok"}


summary_agent = SummaryAgent()
