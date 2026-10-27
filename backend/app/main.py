r"""
FastAPI 主应用入口

启动：
    cd backend
    .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000

注意 --reload-dir app：默认 uvicorn 会监听整个 backend/，
改一下 tests/ 里的脚本就会触发热重启，正在跑的测试会撞上半路重启。
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings

settings = get_settings()

# ---------- 日志 ----------
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG" if settings.debug else "INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> - {message}",
)
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "app_{time:YYYYMMDD}.log",
    rotation="10 MB",
    retention="14 days",
    level="INFO",
    encoding="utf-8",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭钩子"""
    logger.info("=" * 60)
    logger.info("启动 智能知识管理系统 - 后端服务")
    logger.info(f"环境: {settings.environment} | 调试: {settings.debug}")
    logger.info(f"MySQL : {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    logger.info(f"Milvus: {settings.milvus_host}:{settings.milvus_port}")
    logger.info(f"BGE   : {settings.bge_model_path} ({settings.bge_device})")
    logger.info(f"LLM   : {settings.deepseek_model}")
    logger.info("=" * 60)

    # 目录
    settings.ensure_dirs()

    # 数据库建表
    try:
        from app.db.database import init_db

        init_db()
    except Exception as exc:
        logger.error(f"数据库初始化失败: {exc}")
        logger.warning("服务继续启动，但文档相关接口将不可用")

    # Milvus 预热
    try:
        from app.services.milvus_store import milvus_store

        if milvus_store.health():
            logger.info("Milvus 连接正常")
        else:
            logger.warning("Milvus 暂不可用，向量检索接口将报错")
    except Exception as exc:
        logger.warning(f"Milvus 预热失败: {exc}")

    # 后台预热 BGE 模型：不阻塞服务启动，但避免第一次上传/检索时
    # 让用户等上几十秒（导入 transformers + 读 2.2GB 权重）
    _warmup_embedding()

    # 问答缓存预热：把历史问题装进内存 BM25 索引，顺便加载 jieba 词典
    try:
        from app.services.qa_cache_service import kb_signature, qa_cache_service

        qa_cache_service.ensure_index(kb_signature(), force=True)
        logger.info(f"问答缓存就绪：{qa_cache_service.stats()['active_entries']} 条")
    except Exception as exc:
        logger.warning(f"问答缓存预热失败: {exc}")

    # 重排模型（bge-reranker-large）加载要一分多钟，放后台慢慢来；
    # 加载完成前缓存只走词面通道，不会阻塞任何请求
    try:
        from app.services.rerank_service import rerank_service

        rerank_service.warm_async()
        logger.info(f"重排模型开始后台加载（{rerank_service.status()['state']}）")
    except Exception as exc:
        logger.warning(f"重排模型启动失败，问答缓存退化为词面匹配: {exc}")

    logger.success(
        f"服务已就绪 -> http://127.0.0.1:{settings.port}{settings.api_prefix}/docs"
    )
    yield
    logger.info("服务已关闭")


def _warmup_embedding() -> None:
    """在后台线程里预加载嵌入模型"""
    import threading

    def _run() -> None:
        try:
            # 分词词典单独加载一次（约 0.5s），避免第一次提问时卡一下
            from app.utils.chinese_tokenizer import ensure_ready as warm_tokenizer

            warm_tokenizer()
        except Exception as exc:
            logger.warning(f"分词词典预热失败: {exc}")

        try:
            from app.services.embedding_service import embedding_service

            embedding_service.encode("预热")
            logger.success(
                f"嵌入模型预热完成（{embedding_service.device}, {embedding_service.dimension} 维）"
            )
        except Exception as exc:  # 预热失败不影响服务，后续按需加载
            logger.warning(f"嵌入模型预热失败，将在首次调用时重试: {exc}")

    threading.Thread(target=_run, name="bge-warmup", daemon=True).start()


app = FastAPI(
    title="智能知识管理系统",
    description=(
        "基于多 Agent（LangChain）+ BGE-M3 + Milvus 的本地知识库系统。\n\n"
        "Agent 分工：\n"
        "- **IndexAgent** 文档解析与向量入库\n"
        "- **SearchAgent** 语义检索\n"
        "- **QAAgent** RAG 问答\n"
        "- **SummaryAgent** 摘要与关键词\n"
        "- **RelationAgent** 知识图谱\n"
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # 没有 Cookie / 会话认证，不需要放开 credentials；
    # 且 allow_origins="*" 与 allow_credentials=True 同时存在本身就不合规范
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 路由注册 ----------
from app.api import cache, conversations, documents, graph, qa, system  # noqa: E402

app.include_router(system.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)
app.include_router(qa.router, prefix=settings.api_prefix)
app.include_router(graph.router, prefix=settings.api_prefix)
app.include_router(conversations.router, prefix=settings.api_prefix)
app.include_router(cache.router, prefix=settings.api_prefix)


@app.middleware("http")
async def no_store_for_api(request, call_next):
    """
    接口响应一律不缓存

    文档状态是轮询获取的，若被浏览器/代理缓存，前端会一直看到旧状态
    （表现为"上传后一直显示索引中，刷新页面才变"）。
    """
    response = await call_next(request)
    if request.url.path.startswith(settings.api_prefix):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def root():
    return {
        "name": "智能知识管理系统",
        "version": "0.2.0",
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
