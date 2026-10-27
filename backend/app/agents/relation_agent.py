"""
RelationAgent - 知识关联智能体

用文档的「文件名 + 标题 + 摘要 + 正文开头」做向量表征，两两算余弦相似度构建知识图谱：
    节点 = 文档
    边   = 语义相似度超过阈值的文档对

为什么不只用摘要：摘要只有一两百字，跨主题文档的相似度普遍落在 0.3~0.5，
阈值很难选，经常一条边都连不出来。掺入正文片段后区分度明显更好。

除语义相似度外，还补一类「文件名关联」边：
    名字高度相似的两篇文档（"作业2-王小明" 与 "王小明"、"周报-XX" 与 "周报-XX-v2"）
    即使正文完全不同，也往往属于同一个系列，值得在图上连起来。
    这类边用虚线绘制，与语义边区分开。

性能：
    每篇文档的向量按文档缓存（见 `doc_vector_store`），只有新增/改名/重索引的
    文档才会重新编码；两两相似度仍然每次现算（200×200 只要 2ms）。
    文件名相似度用 lru_cache 记住每一对的分数，加文档时只算新涉及的那些对。
"""
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Chunk, Document, DocumentStatus
from app.services import doc_vector_store
from app.utils.name_matcher import NameKey, similarity

settings = get_settings()

# 每篇文档取前几个分块参与表征
SAMPLE_CHUNKS = 3
# 正文参与表征的最大字符数（避免长文档吃掉全部输入长度）
CONTENT_CHARS = 700


@lru_cache(maxsize=100_000)
def _pair_name_score(a: NameKey, b: NameKey) -> float:
    """
    两个名字的相似度（双向取大，且带缓存）

    图谱每次请求都要比 N² 对，纯字符串运算在 200 篇时约 400ms。
    lru_cache 让"加一篇文档"只多算 N 对，其余全部命中；
    缓存键是 NameKey（按归一化文本取哈希），所以改名会自然产生新键，无需手动失效。
    """
    if a.norm > b.norm:  # 固定顺序，让 (A,B) 与 (B,A) 共用一条缓存
        a, b = b, a
    return max(similarity(a, b), similarity(b, a))


class RelationAgent(BaseAgent):
    """文档关系分析智能体"""

    name = "relation_agent"
    description = "计算文档间语义相似度，输出知识图谱节点与边"

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            min_score: 连边阈值，默认 0.45（BGE-M3 余弦，实测 0.4~0.55 较合适）
            max_nodes: 最大节点数，默认 200
            document_id: 若指定，则只输出该文档的邻居子图
            name_min_score: 文件名关联边阈值，0 表示不生成
            force_refresh: 忽略向量缓存，全部重新编码
        """
        min_score = float(kwargs.get("min_score", 0.45))
        max_nodes = int(kwargs.get("max_nodes", 200))
        focus_id = kwargs.get("document_id")
        name_min_score = float(
            kwargs.get("name_min_score", settings.graph_name_edge_threshold)
        )
        force_refresh = bool(kwargs.get("force_refresh", False))

        rows = self._load_documents(max_nodes)
        if len(rows) < 2:
            return {
                "nodes": [self._node(r) for r in rows],
                "edges": [],
                "count": len(rows),
                "min_score": min_score,
                "max_score": None,
                "suggested_score": None,
                "name_edge_count": 0,
                "name_min_score": name_min_score,
                "vector_stats": {
                    "documents": len(rows),
                    "reused": 0,
                    "encoded": 0,
                    "encode_ms": 0,
                },
            }

        # 向量按文档缓存：只有新增/改名/重索引的文档才会真的走 BGE 编码
        vectors, vector_stats = doc_vector_store.get_vectors(
            rows, force_refresh=force_refresh
        )
        sim = vectors @ vectors.T
        max_score = self._max_pair_score(sim)

        edges = self._build_edges(rows, sim, min_score)
        name_edge_count = self._attach_name_edges(
            rows, edges, name_min_score
        )

        if focus_id:
            focus_id_str = str(focus_id)
            edges = [
                e
                for e in edges
                if e["source"] == focus_id_str or e["target"] == focus_id_str
            ]
            keep = {focus_id_str}
            for e in edges:
                keep.add(e["source"])
                keep.add(e["target"])
            rows = [r for r in rows if str(r["id"]) in keep]

        nodes = [self._node(r) for r in rows]

        # 阈值过高时给前端一个建议值，避免用户面对空图不知所措
        suggested = None
        if max_score is not None and max_score > 0:
            suggested = round(max(0.3, max_score - 0.02), 2)

        logger.info(
            f"知识图谱：{len(nodes)} 节点 / {len(edges)} 边 "
            f"(阈值 {min_score}, 最高相似度 {max_score}, 文件名关联 {name_edge_count}, "
            f"向量复用 {vector_stats.get('reused')} / 新编码 {vector_stats.get('encoded')})"
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "count": len(nodes),
            "min_score": min_score,
            "max_score": max_score,
            "suggested_score": suggested,
            "name_edge_count": name_edge_count,
            "name_min_score": name_min_score,
            "vector_stats": vector_stats,
        }

    # ---------- 内部方法 ----------
    @staticmethod
    def _node(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "filename": row.get("filename"),
            "title": row.get("title") or row.get("filename"),
            "category": row["file_type"],
            "value": row.get("chunk_count", 0),
            "summary": row.get("summary") or "",
        }
    @staticmethod
    def _load_documents(limit: int) -> List[Dict[str, Any]]:
        with session_scope() as db:
            rows = (
                db.query(Document)
                .filter(Document.status == DocumentStatus.READY)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .all()
            )
            if not rows:
                return []

            doc_ids = [r.id for r in rows]
            # 每篇取正文开头几个分块，用于增强表征
            chunk_rows = (
                db.query(Chunk.document_id, Chunk.content)
                .filter(Chunk.document_id.in_(doc_ids), Chunk.chunk_index < SAMPLE_CHUNKS)
                .order_by(Chunk.document_id, Chunk.chunk_index)
                .all()
            )

            content_map: Dict[int, str] = {}
            for doc_id, content in chunk_rows:
                if len(content_map.get(doc_id, "")) >= CONTENT_CHARS:
                    continue
                content_map[doc_id] = (content_map.get(doc_id, "") + "\n" + (content or ""))[
                    :CONTENT_CHARS
                ]

            return [
                {
                    "id": r.id,
                    "name": (r.title or r.filename)[:40],
                    "title": r.title,
                    "filename": r.filename,
                    "file_type": r.file_type or "other",
                    "summary": (r.summary or "")[:400],
                    "chunk_count": r.chunk_count,
                    "content": content_map.get(r.id, ""),
                }
                for r in rows
            ]

    @staticmethod
    def _embed_documents(rows: List[Dict[str, Any]]) -> np.ndarray:
        """兼容旧调用：直接走带缓存的向量服务"""
        matrix, _stats = doc_vector_store.get_vectors(rows)
        return matrix

    @staticmethod
    def _max_pair_score(sim: np.ndarray) -> float:
        """最大两两相似度（排除对角线）"""
        n = sim.shape[0]
        if n < 2:
            return 0.0
        masked = sim.copy()
        np.fill_diagonal(masked, -np.inf)
        return round(float(masked.max()), 3)

    @staticmethod
    def _build_edges(
        rows: List[Dict[str, Any]], sim: np.ndarray, min_score: float
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim[i, j])
                if score >= min_score:
                    edges.append(
                        {
                            "source": str(rows[i]["id"]),
                            "target": str(rows[j]["id"]),
                            "value": round(score, 3),
                            "lineStyle": {
                                "width": round(min(4.0, 1.0 + (score - min_score) * 10), 2)
                            },
                        }
                    )
        # 强关联排在前面，便于前端做分层
        edges.sort(key=lambda e: e["value"], reverse=True)
        return edges

    @staticmethod
    def _attach_name_edges(
        rows: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        min_name_score: float,
    ) -> int:
        """
        追加「文件名关联」边，返回新增条数

        名字高度相似但语义相似度不高的文档对（同一作业的不同版本、
        同一系列的第 N 期），在图上应该连起来，否则看起来毫不相干。
        这类边用虚线 + 独立图例，不污染语义相似度的分布。
        """
        if min_name_score <= 0:
            return 0

        existing = {
            frozenset((e["source"], e["target"])): e for e in edges
        }
        # 每篇文档的名字只解析一次（原来每对都要重新构造两个 NameKey，
        # 200 篇时那是 4 万次正则+切分，白白花掉 400ms）
        keys = [NameKey(r.get("filename") or r.get("name")) for r in rows]
        added = 0
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                pair = frozenset((str(rows[i]["id"]), str(rows[j]["id"])))
                name_score = _pair_name_score(keys[i], keys[j])
                if name_score < min_name_score:
                    continue

                if pair in existing:
                    # 语义边已存在，补上名称相似度，并把线改成虚线：
                    # 凡是"名字也对得上"的关联都应该是虚线，用户一眼能看出来
                    edge = existing[pair]
                    edge["name_score"] = round(name_score, 3)
                    edge["relation"] = "both"
                    line_style = dict(edge.get("lineStyle") or {})
                    line_style["type"] = "dashed"
                    edge["lineStyle"] = line_style
                    continue

                edge = {
                    "source": str(rows[i]["id"]),
                    "target": str(rows[j]["id"]),
                    "value": round(name_score, 3),
                    "name_score": round(name_score, 3),
                    "relation": "filename",
                    "lineStyle": {
                        "width": 1.4,
                        "type": "dashed",
                        "opacity": 0.75,
                        "color": "#94a3b8",
                    },
                }
                edges.append(edge)
                existing[pair] = edge
                added += 1

        edges.sort(key=lambda e: e.get("value", 0), reverse=True)
        return added

    def health(self) -> Dict[str, Any]:
        return {"agent": self.name, "status": "ok"}


relation_agent = RelationAgent()
