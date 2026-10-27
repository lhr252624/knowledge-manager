"""文档相关 API"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from loguru import logger

from app.api.schemas import (
    DocumentListResponse,
    DocumentUpdateRequest,
    MessageResponse,
    UploadResponse,
)
from app.config import get_settings
from app.db.database import session_scope
from app.services import document_service
from app.services.document_parser import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/documents", tags=["文档管理"])
settings = get_settings()

MAX_UPLOAD_MB = 50


def _discard_upload(file_path: Path) -> None:
    """删除没有被任何文档记录引用的上传文件

    save_upload() 每次都按 UUID 现写一个新文件，一旦索引流程放弃它
    （内容重复被去重跳过、或解析失败没建记录），这个文件就永远不会
    再被引用，留在磁盘上只是占空间。删除前再查一次库，避免误删仍被
    记录引用（索引中途失败但记录已落库）的文件。
    """
    try:
        path = Path(file_path)
        if not path.is_file():
            return
        with session_scope() as db:
            from app.models.document import Document

            referenced = (
                db.query(Document)
                .filter(Document.file_path.like(f"%{path.name}"))
                .first()
            )
        if referenced is not None:
            return
        path.unlink()
        logger.info(f"已清理无用上传文件: {path.name}")
    except Exception as exc:  # 清理失败不影响主流程
        logger.warning(f"清理上传文件失败: {file_path} -> {exc}")


def _run_index(
    file_path: Path,
    generate_summary: bool = True,
    original_filename: str | None = None,
    discardable: bool = False,
) -> None:
    """后台索引任务

    discardable：这个文件是不是"索引放弃后可以删"的一次性副本。
        上传接口传 True（save_upload 现写的副本，没有记录引用就是垃圾）；
        重建索引传 False（磁盘上那份就是唯一原件，任何情况下都不能删）。
    """
    try:
        result = document_service.index_file(
            file_path,
            generate_summary=generate_summary,
            original_filename=original_filename,
        )
    except Exception as exc:
        logger.exception(f"后台索引失败: {original_filename or file_path.name} -> {exc}")
        # 解析失败时通常连文档记录都没有，文件留在磁盘上也看不到、用不上。
        # 但重建索引的源文件是用户唯一的一份，绝不能删。
        if discardable:
            _discard_upload(file_path)
        return

    if result.get("duplicated"):
        if discardable:
            _discard_upload(file_path)
        logger.info(
            f"内容重复，复用已有文档 id={result.get('document_id')}："
            f"{original_filename or file_path.name}"
        )


@router.post("/upload", response_model=UploadResponse, summary="上传并索引文档")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="待上传文件"),
    generate_summary: bool = Query(True, description="是否生成 AI 摘要"),
):
    """上传文档后立即返回，索引在后台进行，可轮询文档状态"""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}，支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    saved_path = document_service.save_upload(filename, data)
    background_tasks.add_task(
        _run_index,
        saved_path,
        generate_summary,
        filename,
        discardable=True,
    )

    return UploadResponse(
        success=True,
        message="上传成功，正在后台建立索引",
        filename=filename,
    )


@router.get("", response_model=DocumentListResponse, summary="文档列表")
def list_documents(
    status: Optional[str] = Query(None, description="按状态筛选"),
    keyword: Optional[str] = Query(None, description="按名称/标题模糊搜索"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    return document_service.list_documents(
        status=status, keyword=keyword, page=page, size=size
    )


@router.get("/{document_id}", summary="文档详情")
def get_document(
    document_id: int,
    with_chunks: bool = Query(False, description="是否返回分块内容"),
):
    data = document_service.get_document(document_id, with_chunks=with_chunks)
    if data is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return data


@router.get("/{document_id}/content", summary="文档正文预览")
def get_document_content(
    document_id: int,
    max_chars: int = Query(20000, ge=500, le=200000),
):
    data = document_service.get_document(document_id)
    if data is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document_id": document_id,
        "title": data.get("title"),
        "content": document_service.get_document_content(document_id, max_chars),
    }


@router.get("/{document_id}/summary-trace", summary="摘要 Agent 的决策轨迹")
def get_summary_trace(document_id: int):
    """
    返回摘要生成时 ReAct 循环的每一步：调了哪个工具、看到什么、累计读了多少字

    存在的意义：agent 的行为不可复现，必须能复盘"它到底读了哪几段"，
    否则摘要错了都没法定位是没读到，还是读到了却理解错。
    """
    import json

    from app.db.database import session_scope
    from app.models.document import Document

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        raw = doc.summary_trace
        summary = doc.summary
        keywords = doc.keywords

    trace = []
    if raw:
        try:
            trace = json.loads(raw)
        except json.JSONDecodeError:
            trace = []

    return {
        "document_id": document_id,
        "summary": summary,
        "keywords": json.loads(keywords) if keywords else [],
        "mode": "react" if trace else "single",
        "steps": len(trace),
        "trace": trace,
    }


@router.patch("/{document_id}", summary="修改文件名或标题")
def update_document(document_id: int, req: DocumentUpdateRequest):
    """
    改文件名 / 标题

    文件名参与检索（向量里拼了名字，另有独立的名称匹配通道），
    所以改动后会立即重算该文档的向量，改完马上就能按新名字搜到。
    """
    data = document_service.rename_document(
        document_id, filename=req.filename, title=req.title
    )
    if data is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return data


@router.delete("/{document_id}", response_model=MessageResponse, summary="删除文档")
def delete_document(
    document_id: int,
    remove_file: bool = Query(True, description="是否同时删除磁盘文件"),
):
    ok = document_service.delete_document(document_id, remove_file=remove_file)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return MessageResponse(success=True, message="文档已删除")


@router.post("/{document_id}/reindex", response_model=MessageResponse, summary="重新索引")
def reindex_document(document_id: int, background_tasks: BackgroundTasks):
    """
    重新索引

    注意：必须先把文档记录一起删掉。索引流程按文件哈希去重，
    若保留原记录，重建时会被判定为"已存在"而直接跳过。
    磁盘上的源文件保留不动，因此重建后仍是同一份内容。
    """
    with session_scope() as db:
        from app.models.document import Document

        doc = db.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        file_path = Path(doc.file_path)
        original_filename = doc.filename

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"源文件已丢失: {file_path}")

    # 1. 清掉向量
    from app.services.milvus_store import milvus_store

    try:
        milvus_store.delete_by_document(document_id)
    except Exception as exc:
        logger.warning(f"清理旧向量失败: {exc}")

    # 2. 删掉文档记录（级联删除分块），让去重不会命中自己
    with session_scope() as db:
        from app.models.document import Document

        doc = db.get(Document, document_id)
        if doc is not None:
            db.delete(doc)

    # 3. 重新走一遍索引
    background_tasks.add_task(_run_index, file_path, True, original_filename)
    return MessageResponse(success=True, message="已提交重新索引任务")
