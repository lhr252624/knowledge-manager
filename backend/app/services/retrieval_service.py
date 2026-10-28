"""
混合检索服务

语义检索一直是"纯向量"的：BGE-M3 只看正文，文件名和标题根本不参与。
结果是用户输入"作业2 王小明"这种**文档名**时，如果正文里没出现这几个字，
一篇明明存在的文档却搜不出来 —— 名字是用户脑子里最直接的索引，不能丢。

这里做两路召回再融合：

    向量通道   Milvus + BGE-M3，负责"意思相近"
    名称通道   MySQL 里的 filename / title 做字符串匹配，负责"就是这篇"

融合规则：
    1. 名称通道先算出候选文档（含相似度），相似度 ≥ 阈值视为命中
    2. 命中文档若没被向量通道召回，就用它的 document_id 再走一次
       带过滤条件的向量检索，拿到"这篇文档里最贴题的片段"（而不是瞎给第一条）
    3. 最终分数 = 向量分 + 名称相似度 × filename_boost（上限 1.0）
    4. 名称强命中的文档保底占 1 个位置，避免被一堆弱相关的正文片段挤掉
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json

from loguru import logger

from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Document, DocumentStatus
from app.services.milvus_store import milvus_store
from app.services.rerank_service import rerank_service
from app.utils.name_matcher import NameKey, match_score

settings = get_settings()

# 名称匹配最多扫描多少篇文档（名称比较是纯字符串运算，几百篇毫无压力）
MAX_NAME_SCAN = 2000


def _hit_key(hit: Dict[str, Any]) -> Tuple[int, int]:
    return int(hit["document_id"]), int(hit["chunk_index"])


_DUP_CACHE: Dict[str, Any] = {"mtime": None, "map": {}}


def duplicate_map() -> Dict[int, int]:
    """
    读取"重复文档 -> 代表文档"映射（映射文件变了自动重载）

    只处理"同一份文件传了两遍"这种真重复；同主题但内容不同的笔记不该被并，
    所以阈值取 0.6（实测不同章节的复习笔记 Jaccard 只有 0.15~0.39）。
    """
    if not settings.retrieval_dedup_enabled:
        return {}
    path = Path(settings.retrieval_dedup_file)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _DUP_CACHE["mtime"] == mtime:
        return _DUP_CACHE["map"]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = {int(k): int(v) for k, v in (payload.get("duplicates") or {}).items()}
    except Exception as exc:  # 映射坏了不影响检索
        logger.warning(f"读取重复文档映射失败: {exc}")
        mapping = {}

    _DUP_CACHE["mtime"] = mtime
    _DUP_CACHE["map"] = mapping
    logger.info(f"重复文档映射已加载：{len(mapping)} 篇被降权")
    return mapping


def _final_key(hit: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    最终排序键：重排被采纳时按重排分，否则按融合分

    不能混着重排分和自然分布分数一起比：两者量纲不同，混排会让
    "没参加重排的低分片段"莫名其妙跑到前面。
    """
    if not hit.get("rerank_applied"):
        return (0, 0.0, float(hit.get("score") or 0))
    return (1, float(hit.get("rerank_score") or 0), float(hit.get("score") or 0))


def _rerank(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    用交叉编码器把候选按"与问题的相关性"重排

    向量召回是双塔：问题与段落各自编码，最后只做一次点积——它知道"两边像不像"，
    但看不到词与词之间的关系，也分不清"提到了同一个词"和"确实在回答这个问题"。
    交叉编码器把 (问题, 段落) 拼起来过一遍，判定质量高一个档，代价是每条一次前向
    （约 40ms），所以只重排前 N 条，剩下的沿用向量排序。
    """
    if not settings.retrieval_rerank_enabled or len(hits) < 2:
        return hits
    if not rerank_service.ready:
        logger.debug("重排模型未就绪，本次跳过段落重排")
        return hits

    n = min(len(hits), max(1, int(settings.retrieval_rerank_candidates)))
    head, tail = hits[:n], hits[n:]
    try:
        scores = rerank_service.score_pairs(
            [(query, (h.get("content") or "")[:900]) for h in head],
            batch_size=int(settings.retrieval_rerank_batch_size),
        )
    except Exception as exc:  # 显存不足/模型异常时退回向量排序，不能让检索整体失败
        logger.warning(f"段落重排失败，退回向量排序: {exc}")
        return hits
    if len(scores) != len(head):  # 推理异常时保持原序，不要清空结果
        logger.warning("重排返回条数与候选不一致，跳过重排")
        return hits

    for hit, score in zip(head, scores):
        hit["rerank_score"] = round(float(score), 4)

    # 重排分数本身是"这条资料在不在回答这个问题"的判据，但它对**问得太泛**的问题
    # 会集体给低分（实测"介绍一下agent"最高只有 0.10，而具体问题能到 0.8~0.97）。
    # 这种情况按重排分排序会把原本靠前的正确片段压到后面，所以置信不足时保持向量顺序。
    best = max(scores)
    if best < float(settings.retrieval_rerank_min_confident):
        logger.debug(
            f"重排最高分仅 {best:.3f}（< {settings.retrieval_rerank_min_confident}），"
            f"置信不足，保持向量顺序"
        )
        return hits

    for hit in head:
        hit["rerank_applied"] = True
    head.sort(key=lambda h: (-float(h["rerank_score"]), -float(h.get("score") or 0)))
    logger.debug(
        f"重排 {len(head)} 条：最高 {head[0]['rerank_score']:.3f} / "
        f"最低 {head[-1]['rerank_score']:.3f}"
    )
    return head + tail


def name_matches(
    query: str,
    document_ids: Optional[Iterable[int]] = None,
    threshold: Optional[float] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    找出名字（文件名或标题）与提问匹配的文档

    Returns:
        {document_id: {"score": float, "field": "filename"/"title", "text": str}}
    """
    query = (query or "").strip()
    if not query:
        return {}

    threshold = (
        settings.filename_match_threshold if threshold is None else threshold
    )
    key = NameKey(query)

    with session_scope() as db:
        q = db.query(Document.id, Document.filename, Document.title).filter(
            Document.status == DocumentStatus.READY
        )
        if document_ids:
            ids = [int(i) for i in document_ids]
            q = q.filter(Document.id.in_(ids))
        rows = q.limit(MAX_NAME_SCAN).all()

    matched: Dict[int, Dict[str, Any]] = {}
    for doc_id, filename, title in rows:
        score, field = match_score(key, filename, title)
        if score >= threshold:
            matched[int(doc_id)] = {
                "score": round(float(score), 3),
                "field": field,
                "text": (filename if field == "filename" else title) or filename or "",
            }
    return matched


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    document_ids: Optional[Iterable[int]] = None,
    use_name_match: bool = True,
) -> Dict[str, Any]:
    """
    混合检索入口

    Returns:
        {
          "query": str,
          "hits": [片段命中，按 score 降序],
          "name_matches": {doc_id: {...}},   # 名称命中的文档
        }
    """
    query = (query or "").strip()
    top_k = max(1, int(top_k or settings.search_top_k))
    if not query:
        return {"query": query, "hits": [], "name_matches": {}}

    # ---------- 1. 名称通道 ----------
    matched = name_matches(query, document_ids) if use_name_match else {}

    # ---------- 2. 向量通道 ----------
    # 多召回一些，融合后再裁剪
    vector_hits = milvus_store.search(
        query=query, top_k=max(top_k * 3, 12), document_ids=document_ids
    )
    from_vector = {_hit_key(h) for h in vector_hits}
    recalled_docs = {h["document_id"] for h in vector_hits}

    # ---------- 3. 名称命中文档补召回 ----------
    missing = sorted(
        (d for d in matched if d not in recalled_docs),
        key=lambda d: matched[d]["score"],
        reverse=True,
    )
    if missing:
        try:
            extra = milvus_store.search(
                query=query,
                top_k=max(top_k * 2, 8),
                document_ids=missing,
                min_score=0.0,  # 名称命中优先，不因分数过低被丢弃
            )
        except Exception as exc:  # 补召回失败不应影响主链路
            logger.warning(f"名称命中文档补召回失败: {exc}")
            extra = []
        for hit in extra:
            if _hit_key(hit) not in from_vector:
                from_vector.add(_hit_key(hit))
                hit["_supplement"] = True
                vector_hits.append(hit)

    # ---------- 4. 融合打分 ----------
    boost = float(settings.filename_boost)
    for hit in vector_hits:
        info = matched.get(int(hit["document_id"]))
        name_score = float(info["score"]) if info else 0.0
        supplemented = bool(hit.pop("_supplement", False))
        hit["vector_score"] = round(float(hit["score"]), 4)
        hit["name_score"] = name_score
        hit["name_field"] = info["field"] if info else None
        hit["name_text"] = info["text"] if info else None
        hit["score"] = round(min(1.0, hit["vector_score"] + boost * name_score), 4)
        if name_score >= settings.filename_match_threshold:
            hit["match_type"] = "filename" if supplemented else "both"
        else:
            hit["match_type"] = "vector"

    # 同一文档同一分块只保留一条（两路召回可能撞车）
    deduped: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for hit in vector_hits:
        k = _hit_key(hit)
        if k not in deduped or hit["score"] > deduped[k]["score"]:
            deduped[k] = hit
    ranked = sorted(deduped.values(), key=lambda h: h["score"], reverse=True)

    # 融合打分算完后，把"重复上传的副本"压下去，避免同一篇文档占满上下文
    dups = duplicate_map()
    if dups:
        penalty = float(settings.retrieval_dedup_penalty)
        for hit in ranked:
            rep = dups.get(int(hit["document_id"]))
            if rep is not None:
                hit["duplicate_of"] = rep
                hit["score"] = round(float(hit["score"]) * penalty, 4)
        ranked.sort(key=lambda h: h["score"], reverse=True)

    # ---------- 4.5 交叉编码器重排 ----------
    ranked = _rerank(query, ranked)

    # ---------- 5. 名称强命中文档保底 ----------
    strong_docs = [
        d
        for d in sorted(matched, key=lambda d: matched[d]["score"], reverse=True)
        if matched[d]["score"] >= settings.filename_strong_threshold
    ][:2]

    final: List[Dict[str, Any]] = []
    used: set = set()
    for doc_id in strong_docs:
        candidate = next(
            (h for h in ranked if int(h["document_id"]) == doc_id), None
        )
        if candidate is not None and _hit_key(candidate) not in used:
            final.append(candidate)
            used.add(_hit_key(candidate))

    for hit in ranked:
        if len(final) >= top_k:
            break
        k = _hit_key(hit)
        if k in used:
            continue
        final.append(hit)
        used.add(k)

    final.sort(key=_final_key, reverse=True)

    if matched:
        summary = ", ".join(
            f"doc{d}={matched[d]['score']}" for d in list(matched)[:5]
        )
        logger.info(
            f"检索 '{query[:30]}' 名称命中 {len(matched)} 篇 ({summary})，"
            f"最终 {len(final)} 条片段"
        )
    else:
        logger.info(f"检索 '{query[:30]}' 命中 {len(final)} 条片段（无名称命中）")

    return {"query": query, "hits": final, "name_matches": matched}
