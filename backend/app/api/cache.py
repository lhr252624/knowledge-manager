"""问答缓存 API（查看/清理）"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.api.schemas import MessageResponse
from app.db.database import session_scope
from app.models.qa_cache import QACache
from app.services.qa_cache_service import kb_signature, qa_cache_service

router = APIRouter(prefix="/cache", tags=["问答缓存"])


@router.get("/stats", summary="缓存统计")
def cache_stats():
    try:
        return qa_cache_service.stats()
    except Exception as exc:
        logger.exception("读取缓存统计失败")
        raise HTTPException(status_code=500, detail=f"读取缓存统计失败: {exc}") from exc


@router.get("", summary="缓存条目列表")
def list_entries(
    limit: int = Query(50, ge=1, le=500),
    current_only: bool = Query(True, description="只列当前知识库指纹下的条目"),
):
    try:
        with session_scope() as db:
            query = db.query(QACache).order_by(QACache.id.desc())
            if current_only:
                query = query.filter(QACache.kb_signature == kb_signature())
            rows = query.limit(limit).all()
            return {
                "items": [r.to_dict() for r in rows],
                "current_signature": kb_signature(),
            }
    except Exception as exc:
        logger.exception("读取缓存条目失败")
        raise HTTPException(status_code=500, detail=f"读取缓存条目失败: {exc}") from exc


@router.delete("", response_model=MessageResponse, summary="清空缓存")
def clear_cache(
    current_only: bool = Query(False, description="只清当前知识库指纹下的条目"),
):
    signature: Optional[str] = kb_signature() if current_only else None
    removed = qa_cache_service.clear(signature)
    return MessageResponse(success=True, message=f"已清理 {removed} 条缓存")
