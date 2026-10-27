"""
文档表征向量缓存

知识图谱要算「文档两两相似度」，现有做法是每次请求都把全部文档重新编码一遍：

    6 篇   -> BGE 编码 437ms，矩阵 2ms
    200 篇 -> BGE 编码 4774ms，矩阵 2ms

也就是说 **99% 的时间花在"每篇文档自己"的编码上，跟"两两比较"没关系**。
而且图谱页的阈值滑块每拖一下都会重新请求一次接口，等于每拖一下重编码一遍全库。

这里的做法是把编码结果按文档缓存起来（MySQL `doc_vectors`）：

    1. 每篇文档算一个"表征文本"（文件名 + 标题 + 摘要 + 正文片段）
    2. 用它的 sha256 当指纹，连同 document_id、模型标识一起作为命中条件
    3. 命中就直接读回向量，只有指纹变了才重新编码那一篇
    4. 两两相似度仍然每次用 numpy 现算（200×200 只要 2ms）

为什么不去缓存"两两相似度矩阵"（更直观的想法）：
    矩阵本身只要 2ms，存下来省不到时间；但它的失效逻辑很麻烦 ——
    删一篇文档要删掉/改写 N 条记录，改一篇要重算 N 条。
    而向量是"每篇文档一份"，增删改天然只影响它自己，和上面的收益相比几乎零成本。
"""
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from app.config import get_settings
from app.db.database import session_scope
from app.models.doc_vector import DocVector
from app.services.embedding_service import embedding_service
from app.utils.name_matcher import strip_ext

settings = get_settings()

# 模型换了（路径/维度变了）就让全部缓存失效
MODEL_TAG = f"{settings.bge_model_path}|{embedding_service.dimension}"


def build_representation(row: Dict[str, Any]) -> str:
    """
    构造用于表征一篇文档的文本

    文件名排在最前：它往往比机器抽出来的标题更贴近用户认知。
    """
    parts: List[str] = []
    stem = strip_ext(row.get("filename") or "")
    if stem:
        parts.append(stem)
    name = row.get("name")
    if name and name != stem:
        parts.append(name)
    if row.get("summary"):
        parts.append(row["summary"])
    if row.get("content"):
        parts.append(row["content"])
    return "。".join(parts)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def get_vectors(
    rows: List[Dict[str, Any]], force_refresh: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    取这些文档的表征向量（带缓存）

    Args:
        rows: [{"id", "name", "filename", "summary", "content"}, ...]
        force_refresh: 忽略缓存，全部重新编码

    Returns:
        (已归一化的 (N, dim) 矩阵, 统计信息)
    """
    if not rows:
        return np.zeros((0, embedding_service.dimension), dtype=np.float32), {
            "documents": 0,
            "reused": 0,
            "encoded": 0,
            "encode_ms": 0,
            "cached": 0,
        }

    started = time.perf_counter()
    texts = [build_representation(r) for r in rows]
    hashes = [_text_hash(t) for t in texts]
    ids = [int(r["id"]) for r in rows]

    # ---------- 1. 读缓存 ----------
    cached: Dict[int, DocVector] = {}
    if not force_refresh:
        try:
            with session_scope() as db:
                for row in (
                    db.query(DocVector).filter(DocVector.document_id.in_(ids)).all()
                ):
                    cached[int(row.document_id)] = row
        except Exception as exc:  # 缓存读失败就全量重算，不影响正确性
            logger.warning(f"读取文档向量缓存失败，本次全部重算: {exc}")
            cached = {}

    dim = embedding_service.dimension
    vectors: List[Optional[np.ndarray]] = [None] * len(rows)
    reused = 0
    stale: List[int] = []

    for i, doc_id in enumerate(ids):
        row = cached.get(doc_id)
        if (
            row is not None
            and row.model_tag == MODEL_TAG
            and row.text_hash == hashes[i]
            and row.dim == dim
        ):
            try:
                vec = np.frombuffer(row.vector, dtype=np.float32).copy()
                if vec.shape[0] == dim:
                    vectors[i] = vec
                    reused += 1
                    continue
            except Exception:  # 坏数据当作未命中
                pass
        stale.append(i)

    # ---------- 2. 只编码未命中/已变化的 ----------
    encode_ms = 0
    if stale:
        encode_started = time.perf_counter()
        fresh = embedding_service.encode_documents([texts[i] for i in stale])
        encode_ms = int((time.perf_counter() - encode_started) * 1000)
        fresh_arr = np.asarray(fresh, dtype=np.float32)

        try:
            with session_scope() as db:
                existing = {
                    int(r.document_id): r
                    for r in db.query(DocVector)
                    .filter(DocVector.document_id.in_([ids[i] for i in stale]))
                    .all()
                }
                for slot, vec in zip(stale, fresh_arr):
                    doc_id = ids[slot]
                    vectors[slot] = vec
                    payload = {
                        "model_tag": MODEL_TAG,
                        "text_hash": hashes[slot],
                        "dim": int(vec.shape[0]),
                        "vector": vec.astype(np.float32).tobytes(),
                    }
                    row = existing.get(doc_id)
                    if row is None:
                        db.add(DocVector(document_id=doc_id, **payload))
                    else:
                        for key, value in payload.items():
                            setattr(row, key, value)
        except Exception as exc:
            # 写缓存失败不影响本次结果
            logger.warning(f"写入文档向量缓存失败: {exc}")

    matrix = np.stack([v if v is not None else np.zeros(dim, dtype=np.float32) for v in vectors])
    matrix = _normalize(np.asarray(matrix, dtype=np.float32))

    stats = {
        "documents": len(rows),
        "reused": reused,
        "encoded": len(stale),
        "encode_ms": encode_ms,
        "cached": len(cached),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "model": MODEL_TAG,
    }
    if stale:
        logger.info(
            f"文档向量：复用 {reused} 篇 / 新编码 {len(stale)} 篇（{encode_ms}ms）"
        )
    else:
        logger.debug(f"文档向量：全部命中缓存（{reused} 篇，{stats['elapsed_ms']}ms）")
    return matrix, stats


def invalidate(document_id: Optional[int] = None) -> int:
    """删除缓存（按文档，或全部）"""
    try:
        with session_scope() as db:
            query = db.query(DocVector)
            if document_id is not None:
                query = query.filter(DocVector.document_id == int(document_id))
            removed = query.delete(synchronize_session=False)
        logger.info(f"已清理文档向量缓存 {removed} 条")
        return int(removed or 0)
    except Exception as exc:
        logger.warning(f"清理文档向量缓存失败: {exc}")
        return 0


def stats() -> Dict[str, Any]:
    """缓存概况（文档数、占用字节、最近写入时间）"""
    from sqlalchemy import func

    try:
        with session_scope() as db:
            count, size, latest = db.query(
                func.count(DocVector.id),
                func.coalesce(func.sum(DocVector.dim * 4), 0),
                func.max(DocVector.updated_at),
            ).one()
    except Exception as exc:
        logger.warning(f"读取文档向量缓存统计失败: {exc}")
        return {"documents": 0, "bytes": 0, "updated_at": None, "model": MODEL_TAG}
    return {
        "documents": int(count or 0),
        "bytes": int(size or 0),
        "updated_at": latest.isoformat() if hasattr(latest, "isoformat") else str(latest),
        "model": MODEL_TAG,
    }
