"""文档表征向量缓存表

知识图谱的计算里，真正贵的是"把每篇文档编码成 1024 维向量"（BGE-M3，6 篇就要 0.4 秒，
200 篇要 5 秒），而"两两算余弦"只是矩阵乘法（200×200 只要 2ms）。

所以缓存的对象应该是**每篇文档自己的向量**，而不是两两相似度：
    - 文档向量只依赖这篇文档自己（文件名 + 标题 + 摘要 + 正文片段），天然可增量
    - 两两相似度由向量一次矩阵乘出来，缓存它反而要跟着增删改逐行维护，得不偿失

命中条件：document_id 相同 + 模型相同 + 表征文本指纹相同。
换模型、改文件名、重新生成摘要、重新索引正文，都会让指纹变化，从而只重算那一篇。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class DocVector(Base):
    """一篇文档的表征向量"""

    __tablename__ = "doc_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # 模型标识（路径 + 维度），换模型时整体失效
    model_tag: Mapped[str] = mapped_column(String(255), nullable=False)
    # 表征文本的 sha256，文本一变（改名/改摘要/重索引）就重算
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    # float32 原始字节，1024 维 = 4KB
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
