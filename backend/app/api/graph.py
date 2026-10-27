"""知识图谱 API"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.agents.relation_agent import relation_agent

router = APIRouter(prefix="/graph", tags=["知识图谱"])


@router.get("", summary="获取知识图谱")
def get_graph(
    min_score: float = Query(
        0.45, ge=0.0, le=1.0, description="连边相似度阈值（BGE-M3 余弦，建议 0.4~0.55）"
    ),
    max_nodes: int = Query(200, ge=2, le=1000),
    document_id: Optional[int] = Query(None, description="只看某文档的邻居子图"),
    name_min_score: float = Query(
        0.6, ge=0.0, le=1.0, description="文件名关联边阈值，0 表示不生成文件名关联边"
    ),
    force_refresh: bool = Query(
        False, description="忽略文档向量缓存，全部重新编码（换模型或怀疑缓存脏了时用）"
    ),
):
    try:
        return relation_agent.run(
            min_score=min_score,
            max_nodes=max_nodes,
            document_id=document_id,
            name_min_score=name_min_score,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logger.exception("构建知识图谱失败")
        raise HTTPException(status_code=500, detail=f"构建知识图谱失败: {exc}") from exc


@router.get("/vector-cache", summary="文档向量缓存概况")
def vector_cache_stats():
    """图谱背后的向量缓存：多少篇、占多少空间"""
    from app.services import doc_vector_store

    return doc_vector_store.stats()


@router.delete("/vector-cache", summary="清空文档向量缓存")
def clear_vector_cache(document_id: Optional[int] = Query(None, description="只清某一篇")):
    """清掉后下次打开图谱会重新编码（一般不需要，换模型时才会用到）"""
    from app.services import doc_vector_store

    removed = doc_vector_store.invalidate(document_id)
    return {"success": True, "removed": removed}
