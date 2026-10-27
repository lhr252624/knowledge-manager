"""
SearchAgent - 检索智能体

混合检索策略：
    向量召回（Milvus / BGE-M3） + 文件名召回（MySQL filename/title）
    -> 融合打分 -> 按文档聚合 -> 回表补充元数据
"""
from typing import Any, Dict, List

from loguru import logger

from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Document
from app.services.retrieval_service import retrieve

settings = get_settings()


class SearchAgent(BaseAgent):
    """语义检索智能体"""

    name = "search_agent"
    description = "基于向量相似度 + 文件名匹配检索知识库，返回相关片段与来源文档"

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            query: 查询文本（必填）
            top_k: 返回条数，默认取配置
            document_ids: 限定文档范围
            group_by_document: 是否按文档聚合，默认 True
            use_name_match: 是否启用文件名/标题匹配，默认 True
        """
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ValueError("查询内容不能为空")

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        group_by_document = kwargs.get("group_by_document", True)
        use_name_match = kwargs.get("use_name_match", True)

        result = retrieve(
            query=query,
            top_k=top_k,
            document_ids=kwargs.get("document_ids"),
            use_name_match=use_name_match,
        )
        hits = self._enrich(result["hits"])
        doc_meta = self._doc_meta(hits)
        logger.info(f"检索 '{query[:30]}' 命中 {len(hits)} 条片段")

        docs = self._group(hits) if group_by_document else []

        return {
            "query": query,
            "total_chunks": len(hits),
            "chunks": hits,
            "documents": docs,
            "name_matches": self._name_matches(result["name_matches"], doc_meta),
        }

    # ---------- 内部方法 ----------
    def _enrich(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """补充文档元数据（标题、摘要等）"""
        if not hits:
            return []

        meta = self._doc_meta(hits)
        for h in hits:
            m = meta.get(h["document_id"], {})
            h["title"] = m.get("title") or h.get("filename")
            h["filename"] = h.get("filename") or m.get("filename")
            h["file_type"] = m.get("file_type")
            h["document_summary"] = m.get("summary")
            # 片段预览
            content = h.get("content") or ""
            h["preview"] = content[:160].replace("\n", " ")
        return hits

    @staticmethod
    def _doc_meta(hits: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        if not hits:
            return {}
        doc_ids = {h["document_id"] for h in hits}
        meta: Dict[int, Dict[str, Any]] = {}
        with session_scope() as db:
            rows = db.query(Document).filter(Document.id.in_(doc_ids)).all()
            for row in rows:
                # 提前取出属性，避免会话关闭后懒加载
                meta[row.id] = {
                    "title": row.title,
                    "filename": row.filename,
                    "file_type": row.file_type,
                    "summary": row.summary,
                }
        return meta

    @staticmethod
    def _name_matches(
        matched: Dict[int, Dict[str, Any]], meta: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """整理名称命中的文档列表（供前端展示"因为文件名命中而召回"）"""
        out = []
        for doc_id, info in sorted(
            matched.items(), key=lambda kv: kv[1]["score"], reverse=True
        ):
            m = meta.get(doc_id, {})
            out.append(
                {
                    "document_id": doc_id,
                    "score": info["score"],
                    "field": info["field"],
                    "text": info["text"],
                    "title": m.get("title") or m.get("filename"),
                    "filename": m.get("filename"),
                }
            )
        return out

    @staticmethod
    def _group(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按文档聚合，取最高分作为文档相关度"""
        grouped: Dict[int, Dict[str, Any]] = {}
        for h in hits:
            did = h["document_id"]
            if did not in grouped:
                grouped[did] = {
                    "document_id": did,
                    "title": h.get("title"),
                    "filename": h.get("filename"),
                    "file_type": h.get("file_type"),
                    "summary": h.get("document_summary"),
                    "best_score": h["score"],
                    "hit_count": 0,
                    "name_score": h.get("name_score", 0.0),
                    "match_type": h.get("match_type", "vector"),
                    "matched_chunks": [],
                }
            item = grouped[did]
            item["hit_count"] += 1
            item["best_score"] = max(item["best_score"], h["score"])
            if h.get("name_score", 0.0) > item["name_score"]:
                item["name_score"] = h["name_score"]
                item["match_type"] = h.get("match_type", "vector")
            if len(item["matched_chunks"]) < 2:
                item["matched_chunks"].append(
                    {
                        "chunk_index": h["chunk_index"],
                        "preview": h["preview"],
                        "score": h["score"],
                        "vector_score": h.get("vector_score"),
                    }
                )

        return sorted(grouped.values(), key=lambda x: x["best_score"], reverse=True)

    def health(self) -> Dict[str, Any]:
        from app.services.milvus_store import milvus_store

        ok = milvus_store.health()
        return {"agent": self.name, "milvus": ok, "status": "ok" if ok else "down"}


search_agent = SearchAgent()
