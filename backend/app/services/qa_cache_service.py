"""
问答缓存服务

一次问答的成本：一次 BGE 编码 + 一次 Milvus 检索 + 一次 DeepSeek 调用（2~8 秒）。
而重复提问的比例并不低（"下周计划是什么"一天能被问三遍），所以做一层语义缓存。

命中分三层，越往后越贵，但都不会误判：

    ① 精确命中      分词序列 + 否定词 + 数字 完全一致        -> ~0 ms
    ② 词面余弦      BM25 召回 -> IDF 加权余弦 >= 0.85        -> ~2 ms
                    只认"措辞几乎相同"：换标点、换语序、多一个虚词都还能中
    ③ 语义召回+重排  向量召回候选 -> 交叉编码器判定 >= 0.995  -> ~80 ms
                    负责同义改写（"能上传什么格式的文档"），词面完全对不上也能中

三道闸门都是"与"关系里独立的一道，任何一道通过就返回，都不通过才调模型：

    - 为什么不能只用一个阈值：词面余弦的分数在正负例上完全重叠
      （正例 10 分位 0.00，负例最大值 1.00），阈值调到哪都会错一类
    - 为什么语义向量只负责召回、不负责判定：它分不开"前端怎么启动 /
      后端怎么启动"这种最小差异（两者相似度能到 0.95）
    - 为什么判定交给交叉编码器：它把两个问题拼起来看，能发现"是哪个词不一样"
    - 必须校验数字与否定词："2024 年预算"和"2025 年预算"词面几乎相同，答案完全不同

失效策略（比 TTL 更重要）：
    每条缓存记录写入时的知识库指纹（文档数 + 最新更新时间）。
    新增/删除/重建文档后指纹变化，旧缓存自动不参与检索，
    避免"知识库已经更新，回答却还是旧的"。
"""
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from sqlalchemy import func

from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Document
from app.models.qa_cache import QACache
from app.services.bm25 import BM25Index
from app.services.embedding_service import embedding_service
from app.services.rerank_service import rerank_service
from app.utils.chinese_tokenizer import (
    digits,
    has_negation,
    normalize,
    tokenize,
)

settings = get_settings()

# 索引刷新节流：每 3 秒最多查一次库里的行数/最大 id
_REFRESH_INTERVAL = 3.0


def kb_signature() -> str:
    """
    知识库指纹

    文档增删改都会让它变化，从而让旧缓存失效。用 (文档数, 最后更新时间)
    两个指标就够了，没必要为它单独建表。
    """
    try:
        with session_scope() as db:
            count, latest = db.query(
                func.count(Document.id), func.max(Document.updated_at)
            ).one()
    except Exception as exc:  # 数据库抖动时不要让问答整体失败
        logger.warning(f"计算知识库指纹失败，退化为时间戳: {exc}")
        return f"fallback-{int(time.time() // 60)}"

    stamp = latest.isoformat() if hasattr(latest, "isoformat") else str(latest)
    return f"{count or 0}|{stamp}"


def question_fingerprint(text: str) -> str:
    """
    同一分词序列视为同一问题（词序不同、标点不同都不影响）

    注意要把否定词和数字一起编进指纹：停用词过滤会把"不"丢掉，
    "支持 PDF 吗" 和 "不支持 PDF 吗" 的分词结果完全一样，
    只按词序列做指纹会让这两句共用一条缓存。
    """
    tokens = tokenize(text)
    raw = " ".join(tokens) if tokens else normalize(text)
    if has_negation(text):
        raw += " |neg"
    digit_list = sorted(digits(text))
    if digit_list:
        raw += " |d:" + ",".join(digit_list)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def encode_question(text: str) -> Optional[bytes]:
    """问题 -> L2 归一化向量字节；编码失败返回 None（缓存退化为纯词面匹配）"""
    try:
        vec = embedding_service.encode(text, normalize=True)
    except Exception as exc:
        logger.warning(f"问题向量化失败，本次跳过语义通道: {exc}")
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr.tobytes()


def decode_vector(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    try:
        arr = np.frombuffer(blob, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def vector_tag() -> str:
    """向量模型标识：换提供方/换模型/换维度后旧向量自然不匹配"""
    try:
        return embedding_service.model_tag
    except Exception:
        return "unknown"


class QACacheService:
    """进程内索引 + MySQL 持久化的问答缓存"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._index: Optional[BM25Index] = None
        self._by_fingerprint: Dict[str, Dict[str, Any]] = {}
        self._signature: Optional[str] = None
        self._watermark: tuple = (-1, -1)
        self._last_check = 0.0
        # 语义通道：所有条目的向量拼成一个矩阵，检索就是一次矩阵乘法
        self._vectors: Optional[np.ndarray] = None
        self._vector_dim = 0
        # 进程级统计（重启归零，用于看本次运行的效果）
        self.hits = 0
        self.misses = 0
        self.rerank_hits = 0
        self.semantic_hits = 0

    # ---------- 索引维护 ----------
    def _db_watermark(self) -> tuple:
        with session_scope() as db:
            count, max_id = db.query(
                func.count(QACache.id), func.max(QACache.id)
            ).one()
        return int(count or 0), int(max_id or 0)

    def ensure_index(self, signature: str, force: bool = False) -> None:
        """
        按需重建索引

        触发条件：指纹变化、库里的条目数/最大 id 变化、或距上次检查超过 3 秒
        """
        now = time.time()
        if (
            not force
            and self._index is not None
            and self._signature == signature
            and now - self._last_check < _REFRESH_INTERVAL
        ):
            return

        with self._lock:
            now = time.time()
            if (
                not force
                and self._index is not None
                and self._signature == signature
                and now - self._last_check < _REFRESH_INTERVAL
            ):
                return

            try:
                watermark = self._db_watermark()
            except Exception as exc:
                logger.warning(f"读取缓存水位失败，沿用旧索引: {exc}")
                self._last_check = now
                return

            self._last_check = now
            if (
                not force
                and watermark == self._watermark
                and self._signature == signature
            ):
                return

            self._load(signature, watermark)

    def _load(self, signature: str, watermark: tuple) -> None:
        """从 MySQL 读取当前指纹下的有效缓存，构建 BM25 语料 + 向量矩阵"""
        now = datetime.now()
        missing: List[int] = []
        try:
            with session_scope() as db:
                rows = (
                    db.query(QACache)
                    .filter(QACache.kb_signature == signature)
                    .order_by(QACache.id)
                    .all()
                )
                entries = []
                vectors: List[Optional[np.ndarray]] = []
                for row in rows:
                    if row.expires_at is not None and row.expires_at < now:
                        continue
                    try:
                        tokens = tuple(json.loads(row.tokens or "[]"))
                    except (json.JSONDecodeError, TypeError):
                        tokens = tokenize(row.question)
                    entries.append(
                        {
                            "id": row.id,
                            "question": row.question,
                            "tokens": tokens,
                            "answer": row.answer,
                            "citations": _safe_json(row.citations, []),
                            "hit_count": row.hit_count or 0,
                            "created_at": row.created_at,
                            "fingerprint": row.question_hash,
                        }
                    )
                    vec = decode_vector(row.vector)
                    if vec is None:
                        missing.append(len(entries) - 1)
                    vectors.append(vec)
        except Exception as exc:
            logger.warning(f"加载问答缓存失败: {exc}")
            return

        self._entries = entries
        self._index = BM25Index([e["tokens"] for e in entries]) if entries else None
        self._by_fingerprint = {e["fingerprint"]: e for e in entries}
        self._vectors = self._stack_vectors(vectors)
        self._signature = signature
        self._watermark = watermark
        logger.info(
            f"问答缓存索引就绪：{len(entries)} 条（指纹 {signature}，"
            f"其中 {0 if self._vectors is None else len(entries) - len(missing)} 条带向量）"
        )
        if missing:
            self._backfill_vectors([entries[i]["id"] for i in missing[:30]])

    def _stack_vectors(self, vectors: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """把每条向量摞成矩阵；缺失或维度不一致的行留零行（内积恒为 0，不会命中）"""
        dims = [v.size for v in vectors if v is not None]
        if not dims:
            self._vector_dim = 0
            return None
        # 以出现最多的维度为准，换模型后残留的旧维度向量自动作废
        dim = max(set(dims), key=dims.count)
        self._vector_dim = dim
        matrix = np.zeros((len(vectors), dim), dtype=np.float32)
        for i, vec in enumerate(vectors):
            if vec is not None and vec.size == dim:
                matrix[i] = vec
        return matrix

    def _backfill_vectors(self, ids: List[int]) -> None:
        """
        给没有向量的历史条目补向量

        只补一小批（默认 30 条）：首次请求时补太多会让第一问明显变慢。
        剩下的等各自被重新提问时由 store() 覆盖。
        """
        if not ids:
            return
        try:
            with session_scope() as db:
                rows = (
                    db.query(QACache.id, QACache.question)
                    .filter(QACache.id.in_(ids))
                    .all()
                )
                # 先把编码做完（不加锁、不占事务），再统一写回
                encoded = []
                for cache_id, question in rows:
                    blob = encode_question(question)
                    if blob is None:
                        break
                    encoded.append((cache_id, blob))
            if not encoded:
                return

            tag = vector_tag()
            by_id = {e["id"]: i for i, e in enumerate(self._entries)}
            with session_scope() as db:
                for cache_id, blob in encoded:
                    db.query(QACache).filter(QACache.id == cache_id).update(
                        {"vector": blob, "vector_model": tag}
                    )
                    idx = by_id.get(cache_id)
                    vec = decode_vector(blob)
                    if (
                        idx is not None
                        and vec is not None
                        and self._vectors is not None
                        and vec.size == self._vector_dim
                    ):
                        self._vectors[idx] = vec
            logger.info(f"已为 {len(encoded)} 条历史缓存补充问题向量")
        except Exception as exc:
            logger.warning(f"补充历史向量失败: {exc}")

    # ---------- 查询 ----------
    def lookup(self, question: str, signature: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """查缓存（命中返回字典，未命中返回 None）"""
        return self.lookup_ex(question, signature)["hit"]

    def lookup_ex(
        self, question: str, signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        查缓存，同时把"差一点命中"的候选也带出来

        Returns:
            {
              "hit": 命中时是答案字典，否则 None
              "near": 未命中但存在相近候选时 {matched_question, similarity, score}
              "method": exact / lexical / rerank / None
              "elapsed_ms": 本次查询耗时
            }
        """
        started = time.perf_counter()
        result: Dict[str, Any] = {
            "hit": None,
            "near": None,
            "method": None,
            "elapsed_ms": 0.0,
        }

        def finish(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            result["hit"] = payload
            if payload is not None:
                result["method"] = payload.get("method")
                self.hits += 1
            else:
                self.misses += 1
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result

        if not settings.qa_cache_enabled:
            return finish()

        question = (question or "").strip()
        if len(question) < settings.qa_cache_min_question_len:
            return finish()

        signature = signature or kb_signature()
        self.ensure_index(signature)
        if self._index is None or not self._entries:
            return finish()

        tokens = tokenize(question)
        if not tokens:
            return finish()

        # ---- ① 精确命中：分词序列完全一致 ----
        exact = self._by_fingerprint.get(question_fingerprint(question))
        if exact is not None:
            return finish(
                {
                    "cache_id": exact["id"],
                    "answer": exact["answer"],
                    "citations": exact["citations"],
                    "similarity": 1.0,
                    "confidence": 1.0,
                    "score": 1.0,
                    "bm25": None,
                    "matched_question": exact["question"],
                    "exact": True,
                    "method": "exact",
                }
            )

        q_digits = digits(question)
        q_negated = has_negation(question)

        # ---- ② 词面通道：BM25 召回 + IDF 加权余弦 ----
        ranked = self._index.search(
            tokens,
            top_k=settings.qa_cache_top_k,
            temperature=settings.qa_cache_temperature,
        )

        # 候选池：词面相似度与语义相似度各取一份，按较高者排序
        candidates: Dict[int, Dict[str, Any]] = {}
        for idx, bm25_score, similarity, confidence in ranked:
            if similarity <= 0:
                continue
            candidates[idx] = {
                "idx": idx,
                "entry": self._entries[idx],
                "lexical": float(similarity),
                "bm25": float(bm25_score),
                "confidence": float(confidence),
                "semantic": 0.0,
            }

        for _idx, cand in sorted(candidates.items(), key=lambda kv: -kv[1]["lexical"]):
            entry = cand["entry"]
            if cand["lexical"] < settings.qa_cache_similarity_threshold:
                break
            if cand["confidence"] < settings.qa_cache_min_confidence:
                continue
            if not self._guards_ok(entry["question"], q_digits, q_negated):
                continue
            return finish(
                {
                    "cache_id": entry["id"],
                    "answer": entry["answer"],
                    "citations": entry["citations"],
                    "similarity": round(cand["lexical"], 4),
                    "confidence": round(cand["confidence"], 4),
                    "score": round(cand["lexical"], 4),
                    "bm25": round(cand["bm25"], 4),
                    "matched_question": entry["question"],
                    "exact": False,
                    "method": "lexical",
                }
            )

        # ---- ③ 语义通道：向量召回 + 交叉编码器判定 ----
        qvec = decode_vector(encode_question(question))
        if qvec is not None and self._vectors is not None and self._vector_dim:
            if qvec.size == self._vector_dim:
                sims = self._vectors @ qvec
                top = np.argsort(-sims)[: max(1, settings.qa_cache_semantic_top_k)]
                for idx in top:
                    score = float(sims[idx])
                    if score < settings.qa_cache_semantic_recall:
                        continue
                    cand = candidates.get(int(idx))
                    if cand is None:
                        candidates[int(idx)] = {
                            "idx": int(idx),
                            "entry": self._entries[int(idx)],
                            "lexical": 0.0,
                            "bm25": 0.0,
                            "confidence": 0.0,
                            "semantic": score,
                        }
                    else:
                        cand["semantic"] = score

        # 只保留守卫能过的候选
        pool = [
            c
            for c in candidates.values()
            if c["semantic"] >= settings.qa_cache_semantic_recall
            or c["lexical"] >= settings.qa_cache_semantic_recall
        ]
        pool = [
            c
            for c in pool
            if self._guards_ok(c["entry"]["question"], q_digits, q_negated)
        ]
        pool.sort(key=lambda c: max(c["lexical"], c["semantic"]), reverse=True)

        if pool:
            best = pool[0]
            result["near"] = {
                "matched_question": best["entry"]["question"],
                "similarity": round(best["lexical"], 4),
                "score": round(max(best["lexical"], best["semantic"]), 4),
            }

        if pool and rerank_service.ready:
            picked = pool[: max(1, settings.qa_rerank_candidates)]
            pairs = [(c["entry"]["question"], question) for c in picked]
            scores = rerank_service.score_pairs(pairs)
            if scores:
                best_i = int(np.argmax(scores))
                if result["near"] is not None:
                    result["near"]["rerank"] = round(float(scores[best_i]), 4)
                    result["near"]["rerank_question"] = picked[best_i]["entry"]["question"]
                if scores[best_i] >= settings.qa_rerank_threshold:
                    winner = picked[best_i]
                    entry = winner["entry"]
                    self.rerank_hits += 1
                    logger.info(
                        f"重排命中：'{question[:24]}' -> '{entry['question'][:24]}' "
                        f"重排分 {scores[best_i]:.3f}"
                    )
                    return finish(
                        {
                            "cache_id": entry["id"],
                            "answer": entry["answer"],
                            "citations": entry["citations"],
                            "similarity": round(winner["lexical"], 4),
                            "confidence": round(winner["confidence"], 4),
                            "score": round(float(scores[best_i]), 4),
                            "semantic": round(winner["semantic"], 4),
                            "bm25": round(winner["bm25"], 4),
                            "matched_question": entry["question"],
                            "exact": False,
                            "method": "rerank",
                        }
                    )

        logger.debug(
            f"缓存未命中：'{question[:24]}' 最佳词面 "
            f"{pool[0]['lexical'] if pool else 0:.3f} / 最佳语义 "
            f"{pool[0]['semantic'] if pool else 0:.3f} / 重排可用={rerank_service.ready}"
        )
        return finish()

    @staticmethod
    def _guards_ok(
        candidate_question: str, q_digits: set, q_negated: bool
    ) -> bool:
        """数字与否定词必须一致（两条各自独立，任一不符直接排除）"""
        if digits(candidate_question) != q_digits:
            return False
        return has_negation(candidate_question) == q_negated

    # ---------- 写入 ----------
    def store(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        signature: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[int]:
        """写入或更新一条缓存（同一问题只保留最新答案）"""
        if not settings.qa_cache_enabled:
            return None

        question = (question or "").strip()
        answer = (answer or "").strip()
        if len(question) < settings.qa_cache_min_question_len or not answer:
            return None

        signature = signature or kb_signature()
        tokens = tokenize(question)
        fingerprint = question_fingerprint(question)
        # 顺手把问题向量算好存下来：写入只发生一次，代价可以忽略；
        # 检索时就能直接做矩阵乘法，不必对每条历史问题重新编码
        blob = encode_question(question)
        ttl = int(settings.qa_cache_ttl_days)
        expires_at = datetime.now() + timedelta(days=ttl) if ttl > 0 else None

        try:
            with session_scope() as db:
                row = (
                    db.query(QACache)
                    .filter(QACache.question_hash == fingerprint)
                    .first()
                )
                payload = {
                    "question": question,
                    "tokens": json.dumps(list(tokens), ensure_ascii=False),
                    "answer": answer,
                    "citations": json.dumps(citations or [], ensure_ascii=False),
                    "kb_signature": signature,
                    "model": model or settings.deepseek_model,
                    "expires_at": expires_at,
                    "vector": blob,
                    "vector_model": vector_tag() if blob is not None else None,
                }
                if row is None:
                    row = QACache(question_hash=fingerprint, hit_count=0, **payload)
                    db.add(row)
                else:
                    # 同一问题重新回答（例如文档更新后）→ 覆盖答案
                    for key, value in payload.items():
                        setattr(row, key, value)
                db.flush()
                cache_id = row.id
        except Exception as exc:
            logger.warning(f"写入问答缓存失败: {exc}")
            return None

        # 写入后强制刷新，让下一条提问立刻能用上
        self.ensure_index(signature, force=True)
        self._evict()
        return cache_id

    def mark_hit(self, cache_id: int, similarity: Optional[float] = None) -> None:
        """记录一次命中（调用方放在后台任务里，避免拖慢响应）"""
        try:
            with session_scope() as db:
                row = db.get(QACache, cache_id)
                if row is None:
                    return
                row.hit_count = (row.hit_count or 0) + 1
                row.last_hit_at = datetime.now()
                if similarity is not None:
                    row.last_similarity = float(similarity)
        except Exception as exc:
            logger.debug(f"更新缓存命中计数失败: {exc}")

    # ---------- 维护 ----------
    def _evict(self) -> None:
        """先清过期，再按上限淘汰"最少用 + 最久没动"的条目"""
        now = datetime.now()
        try:
            with session_scope() as db:
                db.query(QACache).filter(
                    QACache.expires_at.isnot(None), QACache.expires_at < now
                ).delete(synchronize_session=False)

                total = db.query(func.count(QACache.id)).scalar() or 0
                limit = int(settings.qa_cache_max_entries)
                excess = int(total) - limit
                if excess > 0:
                    stale = (
                        db.query(QACache.id)
                        .order_by(
                            QACache.hit_count.asc(),
                            func.coalesce(QACache.last_hit_at, QACache.created_at).asc(),
                        )
                        .limit(excess)
                        .all()
                    )
                    ids = [row[0] for row in stale]
                    if ids:
                        db.query(QACache).filter(QACache.id.in_(ids)).delete(
                            synchronize_session=False
                        )
                        logger.info(f"问答缓存超出上限，淘汰 {len(ids)} 条")
        except Exception as exc:
            logger.warning(f"清理问答缓存失败: {exc}")

    def clear(self, signature: Optional[str] = None) -> int:
        """清空缓存（可只清某个知识库指纹下的）"""
        try:
            with session_scope() as db:
                query = db.query(QACache)
                if signature:
                    query = query.filter(QACache.kb_signature == signature)
                removed = query.delete(synchronize_session=False)
        except Exception as exc:
            logger.warning(f"清空问答缓存失败: {exc}")
            return 0

        self.ensure_index(kb_signature(), force=True)
        # 清空后本次运行的命中统计也归零，否则界面会显示"0 条缓存 / 命中率 50%"
        self.hits = 0
        self.misses = 0
        logger.info(f"已清理问答缓存 {removed} 条")
        return int(removed or 0)

    # ---------- 统计 ----------
    def stats(self) -> Dict[str, Any]:
        signature = kb_signature()
        self.ensure_index(signature)
        try:
            with session_scope() as db:
                total = int(db.query(func.count(QACache.id)).scalar() or 0)
                total_hits = int(db.query(func.coalesce(func.sum(QACache.hit_count), 0)).scalar() or 0)
                recent = (
                    db.query(QACache)
                    .order_by(QACache.id.desc())
                    .limit(5)
                    .all()
                )
                recent_items = [
                    {
                        "question": r.question[:60],
                        "hit_count": r.hit_count or 0,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in recent
                ]
        except Exception as exc:
            logger.warning(f"读取缓存统计失败: {exc}")
            total = total_hits = 0
            recent_items = []

        session_total = self.hits + self.misses
        return {
            "enabled": bool(settings.qa_cache_enabled),
            "entries": total,
            "active_entries": len(self._entries),
            "vector_entries": 0 if self._vectors is None else int(np.count_nonzero(self._vectors.any(axis=1))),
            "total_hits": total_hits,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "session_hit_rate": round(self.hits / session_total, 3) if session_total else 0.0,
            "rerank_hits": self.rerank_hits,
            "similarity_threshold": settings.qa_cache_similarity_threshold,
            "semantic_recall": settings.qa_cache_semantic_recall,
            "rerank": rerank_service.status(),
            "ttl_days": settings.qa_cache_ttl_days,
            "max_entries": settings.qa_cache_max_entries,
            "kb_signature": signature,
            "recent": recent_items,
        }

    def invalidate_all(self) -> None:
        """强制丢弃内存索引（例如切换知识库后）"""
        with self._lock:
            self._entries = []
            self._index = None
            self._by_fingerprint = {}
            self._vectors = None
            self._vector_dim = 0
            self._signature = None
            self._watermark = (-1, -1)
            self._last_check = 0.0


def _safe_json(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


qa_cache_service = QACacheService()
