"""系统状态 API"""
from fastapi import APIRouter

from app.agents.index_agent import index_agent
from app.agents.qa_agent import qa_agent
from app.agents.search_agent import search_agent
from app.config import get_settings
from app.services.document_service import get_stats
from app.services.embedding_service import embedding_service
from app.services.milvus_store import milvus_store

router = APIRouter(tags=["系统"])
settings = get_settings()


@router.get("/health", summary="健康检查")
def health():
    return {"status": "healthy", "environment": settings.environment}


@router.get("/health/detail", summary="详细健康检查")
def health_detail():
    from app.db.database import check_mysql

    checks = {
        "mysql": check_mysql(),
        "milvus": milvus_store.health(),
        "llm": bool(settings.deepseek_api_key),
    }
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "embedding": {
            "model": settings.bge_model_path,
            "loaded": embedding_service.is_loaded(),
            "device": embedding_service.device,
        },
        "llm_model": settings.deepseek_model,
    }


@router.get("/stats", summary="知识库统计")
def stats():
    return get_stats()


@router.get("/agents", summary="Agent 列表与状态")
def agents():
    return {
        "agents": [
            {
                "name": "index_agent",
                "description": index_agent.description,
                "status": "ready",
            },
            {
                "name": "search_agent",
                "description": search_agent.description,
                "status": "ready",
            },
            {
                "name": "qa_agent",
                "description": qa_agent.description,
                "status": "ready",
            },
            {
                "name": "summary_agent",
                "description": "为文档生成摘要与关键词",
                "status": "ready",
            },
            {
                "name": "relation_agent",
                "description": "构建文档知识图谱",
                "status": "ready",
            },
        ]
    }
