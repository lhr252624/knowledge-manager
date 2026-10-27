"""
历史对话服务

仿 DeepSeek 的侧边栏：会话列表、新建、改名、删除、按会话取消息。

容量上的取舍（单用户本地知识库）：
    - 最多 200 个会话，超出后删最久没动的
    - 每个会话最多 200 条消息，超出后删最早的（只留最近的部分）
    - 追问时只带最近 3 轮进 Prompt，每条约 400 字
      不带更多是因为上下文越长越贵、越容易跑题，而 3 轮足够接住"那第二点呢"这种追问
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import get_settings
from app.db.database import session_scope
from app.models.conversation import Conversation, Message

settings = get_settings()

DEFAULT_TITLE = "新对话"


# ---------- 会话 ----------
def list_conversations(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """会话列表（按最后活动时间倒序），带最后一条消息的预览"""
    limit = limit or settings.history_max_conversations
    limit = max(1, min(int(limit), settings.history_max_conversations))

    with session_scope() as db:
        rows = (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            return []

        ids = [r.id for r in rows]
        # 每个会话最后一条消息作为预览（按 id 倒序取第一个）
        last_rows = (
            db.query(Message)
            .filter(Message.conversation_id.in_(ids))
            .order_by(Message.conversation_id, Message.id.desc())
            .all()
        )
        previews: Dict[int, str] = {}
        for msg in last_rows:
            if msg.conversation_id in previews:
                continue
            previews[msg.conversation_id] = (msg.content or "").replace("\n", " ")[:80]

    return [r.to_dict(preview=previews.get(r.id, "")) for r in rows]


def get_conversation(conversation_id: int, with_messages: bool = True) -> Optional[Dict[str, Any]]:
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            return None
        data = conv.to_dict(with_messages=False)
        if with_messages:
            messages = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.id)
                .all()
            )
            data["messages"] = [m.to_dict() for m in messages]
        return data


def create_conversation(title: Optional[str] = None) -> Dict[str, Any]:
    with session_scope() as db:
        conv = Conversation(title=_clean_title(title) or DEFAULT_TITLE, message_count=0)
        db.add(conv)
        db.flush()
        data = conv.to_dict()
    _enforce_conversation_limit()
    return data


def rename_conversation(conversation_id: int, title: str) -> Optional[Dict[str, Any]]:
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            return None
        conv.title = _clean_title(title)[:200] or DEFAULT_TITLE
        db.flush()
        return conv.to_dict()


def delete_conversation(conversation_id: int) -> bool:
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            return False
        db.delete(conv)
    return True


def clear_conversations() -> int:
    with session_scope() as db:
        removed = db.query(Conversation).delete(synchronize_session=False)
    logger.info(f"已清空历史对话 {removed} 个会话")
    return int(removed or 0)


# ---------- 消息 ----------
def add_message(
    conversation_id: int,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    cached: bool = False,
    similarity: Optional[float] = None,
    latency_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            return None

        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content or "",
            citations=json.dumps(citations or [], ensure_ascii=False) if citations else None,
            cached=bool(cached),
            similarity=similarity,
            latency_ms=latency_ms,
        )
        db.add(msg)

        conv.message_count = (conv.message_count or 0) + 1
        conv.updated_at = datetime.now()
        # 第一条用户消息作为标题
        if role == "user" and (conv.title or DEFAULT_TITLE) == DEFAULT_TITLE:
            conv.title = _clean_title(content)[:200] or DEFAULT_TITLE

        db.flush()
        data = msg.to_dict()

    if role == "user":
        _enforce_message_limit(conversation_id)
    return data


def add_user_and_assistant(
    conversation_id: int,
    question: str,
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    cached: bool = False,
    similarity: Optional[float] = None,
    latency_ms: Optional[int] = None,
) -> None:
    """一次问答落两条消息（用户 + 助手）"""
    add_message(conversation_id, "user", question)
    add_message(
        conversation_id,
        "assistant",
        answer,
        citations=citations,
        cached=cached,
        similarity=similarity,
        latency_ms=latency_ms,
    )


def history_for_prompt(
    conversation_id: Optional[int],
    rounds: Optional[int] = None,
    exclude_last: int = 1,
) -> List[Dict[str, str]]:
    """
    取用于 Prompt 的对话历史

    Args:
        exclude_last: 忽略最后 N 条消息（当前这一问刚存进库里）
    """
    if not conversation_id:
        return []

    rounds = settings.history_context_rounds if rounds is None else rounds
    if rounds <= 0:
        return []

    with session_scope() as db:
        rows = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(rounds * 2 + max(0, exclude_last))
            .all()
        )

    rows = list(reversed(rows))
    if exclude_last:
        rows = rows[:-exclude_last]

    limit = int(settings.history_context_chars)
    history = []
    for msg in rows:
        text = (msg.content or "").strip()
        if not text:
            continue
        if len(text) > limit:
            text = text[:limit] + "…"
        history.append({"role": msg.role, "content": text})
    return history


# ---------- 限额维护 ----------
def _enforce_message_limit(conversation_id: int) -> None:
    limit = int(settings.history_max_messages)
    try:
        with session_scope() as db:
            total = (
                db.query(Message).filter(Message.conversation_id == conversation_id).count()
            )
            if total <= limit:
                return
            stale = (
                db.query(Message.id)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.id)
                .limit(total - limit)
                .all()
            )
            ids = [row[0] for row in stale]
            if ids:
                db.query(Message).filter(Message.id.in_(ids)).delete(
                    synchronize_session=False
                )
            conv = db.get(Conversation, conversation_id)
            if conv is not None:
                conv.message_count = int(
                    db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .count()
                )
            logger.info(f"会话 {conversation_id} 消息超出 {limit} 条，清理 {len(ids)} 条")
    except Exception as exc:
        logger.warning(f"清理会话消息失败: {exc}")


def _enforce_conversation_limit() -> None:
    limit = int(settings.history_max_conversations)
    try:
        with session_scope() as db:
            total = int(db.query(Conversation).count())
            if total <= limit:
                return
            stale = (
                db.query(Conversation.id)
                .order_by(Conversation.updated_at.asc())
                .limit(total - limit)
                .all()
            )
            ids = [row[0] for row in stale]
            if ids:
                db.query(Conversation).filter(Conversation.id.in_(ids)).delete(
                    synchronize_session=False
                )
                logger.info(f"会话数超出 {limit}，清理最旧的 {len(ids)} 个")
    except Exception as exc:
        logger.warning(f"清理会话失败: {exc}")


def _clean_title(text: Optional[str]) -> str:
    if not text:
        return ""
    one_line = " ".join(str(text).split())
    return one_line[:60]
