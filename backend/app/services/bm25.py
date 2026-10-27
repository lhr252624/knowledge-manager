"""
BM25 检索 + 有界相似度 + softmax 置信度

问答缓存要在"历史提过的所有问题"里找最像的一条。这里的分工是：

    1. BM25        从几千条历史问题里粗排，负责"召回"（快、无需向量模型）
    2. 加权余弦     给出 0~1 的相似度，负责"判定"（有界、可解释，85% 阈值就卡这里）
    3. softmax     把 BM25 分数变成"我有多确信是这一条"，作为第二道闸

为什么不能只用 softmax 阈值：
    softmax 的分母是候选集，对整个库做 softmax 时，候选越多最大值越小。
    2000 条缓存时，哪怕完美命中，概率也可能只有 0.07 —— 阈值永远过不了。
    所以 softmax 只在 BM25 召回的 top-K（默认 8 条）上做，
    真正决定"是不是同一个问题"的是加权余弦。
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class BM25Index:
    """
    Okapi BM25 索引（语料 = 历史问题）

    参数字典与 Elasticsearch 的默认值一致：k1=1.5, b=0.75
    """

    def __init__(
        self,
        docs: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs: List[List[str]] = [list(d) for d in docs]
        self.k1 = k1
        self.b = b
        self.n = len(self.docs)

        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        if self.avgdl <= 0:
            self.avgdl = 1.0

        self.tf: List[Counter] = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.df = df

        # BM25 的 idf 恒为正，可以直接拿来当加权余弦的权重
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        # 语料里没出现过的词：按最稀有处理
        self.unseen_idf = math.log(1 + (self.n + 0.5) / 0.5) if self.n else 1.0
        self._norm = [self._doc_norm(doc) for doc in self.docs]

    # ---------- 内部 ----------
    def weight(self, term: str) -> float:
        return self.idf.get(term, self.unseen_idf)

    def _doc_norm(self, doc: Sequence[str]) -> float:
        total = sum(self.weight(t) for t in set(doc))
        return math.sqrt(total) if total > 0 else 0.0

    # ---------- 打分 ----------
    def scores(self, query: Sequence[str]) -> np.ndarray:
        """BM25 分数（未归一化，范围随语料变化）"""
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n or not query:
            return out

        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue  # 语料里没有的词对 BM25 没有贡献
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                dl = self.doc_len[i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += idf * freq * (self.k1 + 1) / denom
        return out

    def similarity(self, index: int, query: Sequence[str]) -> float:
        """
        加权余弦相似度（0~1）

        以 idf 作权重，比"交集比例"更能区分"关键词全中"和"只是碰巧有常见词"。
        完全相同的问题 = 1.0，毫无交集 = 0.0。
        """
        if not query or index < 0 or index >= self.n:
            return 0.0

        q_terms = set(query)
        d_terms = set(self.docs[index])
        if not q_terms or not d_terms:
            return 0.0

        shared = q_terms & d_terms
        if not shared:
            return 0.0

        numerator = sum(self.weight(t) for t in shared)
        q_norm = math.sqrt(sum(self.weight(t) for t in q_terms))
        d_norm = self._norm[index]
        if q_norm <= 0 or d_norm <= 0:
            return 0.0
        return float(min(1.0, numerator / (q_norm * d_norm)))

    def search(
        self,
        query: Sequence[str],
        top_k: int = 8,
        temperature: float = 1.0,
    ) -> List[Tuple[int, float, float, float]]:
        """
        召回 + 打分

        Returns:
            [(语料下标, bm25 分数, 相似度, softmax 置信度), ...]，按相似度降序
        """
        if not self.n or not query:
            return []

        bm25 = self.scores(query)
        # 先按 BM25 粗排取 top-K，softmax 只在 K 条上做
        k = max(1, min(top_k, self.n))
        order = np.argsort(-bm25)[:k]

        picked = [(int(i), float(bm25[i])) for i in order]
        # 候选里 BM25 全为 0（一个字都没撞上）时，softmax 没有意义
        if all(score <= 0 for _, score in picked):
            return [(i, s, 0.0, 0.0) for i, s in picked]

        confidences = softmax([s for _, s in picked], temperature)
        result = [
            (i, s, self.similarity(i, query), float(c))
            for (i, s), c in zip(picked, confidences)
        ]
        result.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return result


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """数值稳定的 softmax"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    t = temperature if temperature > 0 else 1.0
    shifted = (arr - arr.max()) / t
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return exp / total


def cosine_tokens(
    a: Sequence[str], b: Sequence[str], idf: Optional[Dict[str, float]] = None
) -> float:
    """两个词袋的加权余弦（没有现成语料时用的简化版，权重全 1）"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    shared = sa & sb
    if not shared:
        return 0.0
    w = idf or {}
    num = sum(w.get(t, 1.0) for t in shared)
    na = math.sqrt(sum(w.get(t, 1.0) for t in sa))
    nb = math.sqrt(sum(w.get(t, 1.0) for t in sb))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(min(1.0, num / (na * nb)))
