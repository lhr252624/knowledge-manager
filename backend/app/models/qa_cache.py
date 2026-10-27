"""问答缓存表

一次问答要花 2~8 秒和一次 API 费用，而用户其实经常重复问同一件事
（"下周计划是什么"能被问三遍）。这里把「问题 → 答案」存下来，
新提问先按中文分词 + BM25 + 加权余弦找历史上最像的问题，
足够像就直接返回，不再调模型。

为什么存 MySQL 而不是 Redis：
    1. BM25 要扫全量历史问题，无论存哪都得把语料读进内存，
       多一个 Redis 只是多一次网络往返
    2. 缓存条目需要能人工查看/删除（答错了要能清掉），SQL 更方便
    3. 进程内缓存本身是热的，落库只为重启后还在
    （真要多副本部署时，把本表换成 Redis Hash 即可，接口层不变）
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class QACache(Base):
    """问题-答案缓存"""

    __tablename__ = "qa_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 去重键：问题归一化分词后排序拼接的 sha256，同一问题只留一条
    question_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True, comment="问题指纹"
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="原始问题")
    tokens: Mapped[str] = mapped_column(
        Text, nullable=False, comment="分词结果（JSON 数组），避免每次重新分词"
    )

    # 问题向量（float32、L2 归一化后的裸字节，1024 维 = 4KB）。
    # 词面匹配抓不住同义改写，这条通道负责"把可能相关的问题捞出来"，
    # 至于"到底是不是同一个问题"由重排模型判定。
    # 存成 BLOB 而不是 JSON：读写更快，也不会因为浮点格式化丢精度。
    vector: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, comment="问题向量 float32（L2 归一化）"
    )
    vector_model: Mapped[Optional[str]] = mapped_column(
        String(128), comment="生成向量的模型标识，换模型后旧向量自动作废"
    )

    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="模型回答")
    citations: Mapped[str] = mapped_column(Text, comment="引用来源（JSON）")

    # 知识库指纹：文档增删改后所有缓存自动失效，避免拿旧答案回答新知识库
    kb_signature: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True, comment="知识库指纹"
    )
    model: Mapped[str] = mapped_column(String(64), comment="生成答案所用模型")

    hit_count: Mapped[int] = mapped_column(Integer, default=0, comment="被命中次数")
    last_similarity: Mapped[Optional[float]] = mapped_column(
        Float, comment="最近一次命中时的相似度"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, comment="最后命中时间")
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, index=True, comment="过期时间，空表示不过期"
    )

    __table_args__ = (Index("ix_qa_cache_kb_hit", "kb_signature", "hit_count"),)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "hit_count": self.hit_count,
            "kb_signature": self.kb_signature,
            "model": self.model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
