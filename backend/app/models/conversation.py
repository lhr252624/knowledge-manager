"""对话与消息表

仿 DeepSeek 的侧边栏历史对话：
    会话（conversations） 1 ── N 消息（messages）

单用户使用，所以没有 user_id；要做多用户时给 conversations 加一列即可。
消息里存 citations，是为了点开历史会话时还能看到当时的引用来源。
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Conversation(Base):
    """一次连续对话"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(
        String(200), nullable=False, default="新对话", comment="标题，默认取第一个问题"
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, comment="消息条数")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
        comment="最后活动时间，侧边栏按它排序",
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )

    def to_dict(self, with_messages: bool = False, preview: Optional[str] = None) -> dict:
        data = {
            "id": self.id,
            "title": self.title,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if preview is not None:
            data["preview"] = preview
        if with_messages:
            data["messages"] = [m.to_dict() for m in self.messages]
        return data


class Message(Base):
    """会话中的一条消息"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="user / assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[str]] = mapped_column(Text, comment="引用来源（JSON）")

    # 缓存命中的回答标记为 cached，界面可以显示"缓存命中"
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    similarity: Mapped[Optional[float]] = mapped_column(Float, comment="缓存命中时的相似度")
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, comment="耗时毫秒")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_conv_id", "conversation_id", "id"),)

    def to_dict(self) -> dict:
        import json

        citations = []
        if self.citations:
            try:
                citations = json.loads(self.citations)
            except (json.JSONDecodeError, TypeError):
                citations = []
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "citations": citations,
            "cached": bool(self.cached),
            "similarity": self.similarity,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
