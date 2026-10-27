"""文档与分块数据模型"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DocumentStatus:
    """文档处理状态"""

    PENDING = "pending"       # 已入库，待处理
    PARSING = "parsing"       # 解析中
    INDEXING = "indexing"     # 向量化中
    READY = "ready"           # 可检索
    FAILED = "failed"         # 处理失败


class Document(Base):
    """文档主表"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 文件信息
    filename: Mapped[str] = mapped_column(String(255), nullable=False, comment="原始文件名")
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="存储文件名")
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, comment="磁盘绝对路径")
    file_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="扩展名")
    file_size: Mapped[int] = mapped_column(BigInteger, default=0, comment="字节数")
    file_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True, comment="SHA256，用于去重"
    )

    # 内容信息
    title: Mapped[Optional[str]] = mapped_column(String(512), comment="标题")
    summary: Mapped[Optional[str]] = mapped_column(Text, comment="摘要")
    keywords: Mapped[Optional[str]] = mapped_column(Text, comment="关键词，JSON 数组")
    char_count: Mapped[int] = mapped_column(Integer, default=0, comment="正文字符数")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="分块数量")

    # 状态
    status: Mapped[str] = mapped_column(
        String(16), default=DocumentStatus.PENDING, index=True
    )
    error_msg: Mapped[Optional[str]] = mapped_column(Text, comment="失败原因")

    # 时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self, with_summary: bool = True) -> dict:
        data = {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "title": self.title,
            "char_count": self.char_count,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error_msg": self.error_msg,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if with_summary:
            data["summary"] = self.summary
        return data


class Chunk(Base):
    """文档分块表"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="块序号")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="块文本")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    milvus_pk: Mapped[Optional[int]] = mapped_column(
        BigInteger, index=True, comment="Milvus 主键"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_chunk", "document_id", "chunk_index"),
    )
