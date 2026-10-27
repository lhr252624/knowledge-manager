"""历史对话 API（仿 DeepSeek 侧边栏）"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.api.schemas import ConversationCreate, ConversationRename, MessageResponse
from app.services import conversation_service

router = APIRouter(prefix="/conversations", tags=["历史对话"])


@router.get("", summary="会话列表")
def list_conversations(
    limit: Optional[int] = Query(None, ge=1, le=500, description="返回条数上限"),
):
    try:
        return {"items": conversation_service.list_conversations(limit)}
    except Exception as exc:
        logger.exception("读取会话列表失败")
        raise HTTPException(status_code=500, detail=f"读取会话列表失败: {exc}") from exc


@router.post("", summary="新建会话")
def create_conversation(req: Optional[ConversationCreate] = None):
    title = req.title if req else None
    return conversation_service.create_conversation(title)


@router.get("/{conversation_id}", summary="会话详情（含消息）")
def get_conversation(
    conversation_id: int,
    with_messages: bool = Query(True),
):
    data = conversation_service.get_conversation(
        conversation_id, with_messages=with_messages
    )
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return data


@router.patch("/{conversation_id}", summary="重命名会话")
def rename_conversation(conversation_id: int, req: ConversationRename):
    data = conversation_service.rename_conversation(conversation_id, req.title)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return data


@router.delete("/{conversation_id}", response_model=MessageResponse, summary="删除会话")
def delete_conversation(conversation_id: int):
    ok = conversation_service.delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return MessageResponse(success=True, message="会话已删除")


@router.delete("", response_model=MessageResponse, summary="清空全部会话")
def clear_conversations():
    removed = conversation_service.clear_conversations()
    return MessageResponse(success=True, message=f"已清空 {removed} 个会话")
