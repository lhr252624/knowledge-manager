"""
文档业务服务

负责文件落盘、数据库 CRUD 与索引任务调度。
"""
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import func

from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Chunk, Document, DocumentStatus

settings = get_settings()


# ---------- 文件保存 ----------
def save_upload(filename: str, data: bytes) -> Path:
    """保存上传文件到 upload_path，返回磁盘路径

    磁盘上用 UUID 命名避免重名与非法字符冲突；原始文件名由调用方
    通过 index_file(original_filename=...) 传给索引流程写入数据库。
    """
    settings.ensure_dirs()
    ext = Path(filename).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    target = Path(settings.upload_path) / stored_name
    target.write_bytes(data)
    logger.info(f"上传文件已保存: {target} ({len(data)} 字节) 原名={filename}")
    return target


# ---------- 索引任务 ----------
def index_file(
    path: Path,
    generate_summary: bool = True,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """执行索引（供后台任务调用）"""
    from app.agents.index_agent import index_agent

    return index_agent.run(
        file_path=str(path),
        generate_summary=generate_summary,
        original_filename=original_filename,
    )


# ---------- 重命名 / 重算向量 ----------
def reembed_document(document_id: int) -> int:
    """
    重算某文档的分块向量

    文件名或标题改动后必须重算：向量里拼了名字前缀（见 build_embed_text），
    不重算的话"按新名字搜"依然搜不到，属于改了等于没改。
    """
    from app.services.milvus_store import milvus_store

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return 0
        filename = doc.filename
        title = doc.title
        rows = (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .all()
        )
        payload = [
            {"chunk_id": c.id, "chunk_index": c.chunk_index, "content": c.content}
            for c in rows
        ]

    if not payload:
        return 0

    milvus_store.delete_by_document(document_id)
    pks = milvus_store.insert_chunks(
        document_id=document_id, filename=filename, chunks=payload, title=title
    )

    with session_scope() as db:
        for item, pk in zip(payload, pks):
            chunk = db.get(Chunk, item["chunk_id"])
            if chunk is not None:
                chunk.milvus_pk = pk
    logger.info(f"文档 {document_id} 向量已重算（{len(pks)} 个分块）")
    return len(pks)


def rename_document(
    document_id: int,
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    改文件名 / 标题

    历史文档的原始文件名在早期版本里丢了（入库时写成了磁盘上的 UUID），
    标题是机器从正文第一行抽的，未必等于用户心里的文件名。
    这里让用户能直接改正，改完立刻重算向量，按新名字就能搜到。
    """
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return None

        changed = False
        new_filename = (filename or "").strip()
        if new_filename and new_filename != doc.filename:
            doc.filename = new_filename[:255]
            changed = True

        if title is not None:
            new_title = title.strip()[:512]
            if new_title != (doc.title or ""):
                doc.title = new_title or None
                changed = True

    if changed:
        try:
            reembed_document(document_id)
        except Exception as exc:  # 重算失败不回滚改名，但要记日志
            logger.warning(f"文档 {document_id} 改名后重算向量失败: {exc}")

    return get_document(document_id)


# ---------- 查询 ----------
def list_documents(
    status: Optional[str] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    size: int = 20,
) -> Dict[str, Any]:
    page = max(1, page)
    size = min(max(1, size), 100)

    with session_scope() as db:
        query = db.query(Document)
        if status:
            query = query.filter(Document.status == status)
        if keyword:
            like = f"%{keyword}%"
            query = query.filter(
                (Document.filename.like(like))
                | (Document.title.like(like))
                | (Document.summary.like(like))
            )

        total = query.count()
        rows = (
            query.order_by(Document.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        items = [r.to_dict() for r in rows]

    return {"total": total, "page": page, "size": size, "items": items}


def get_document(document_id: int, with_chunks: bool = False) -> Optional[Dict[str, Any]]:
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return None
        data = doc.to_dict()
        data["keywords"] = _parse_keywords(doc.keywords)

        if with_chunks:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            data["chunks"] = [
                {
                    "id": c.id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "char_count": c.char_count,
                }
                for c in chunks
            ]
        return data


def get_document_content(document_id: int, max_chars: int = 20000) -> str:
    """取回文档正文（由分块拼接），用于详情预览"""
    with session_scope() as db:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .all()
        )
    text = "\n\n".join(c.content for c in chunks)
    return text[:max_chars]


# ---------- 删除 ----------
def delete_document(document_id: int, remove_file: bool = True) -> bool:
    """删除文档：Milvus 向量 + MySQL 记录 + 磁盘文件"""
    from app.services.milvus_store import milvus_store

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return False
        file_path = Path(doc.file_path)
        stored = doc.filename

    # 1. 删除向量
    try:
        milvus_store.delete_by_document(document_id)
    except Exception as exc:
        logger.warning(f"删除 Milvus 向量失败（继续删除数据库记录）: {exc}")

    # 2. 删除数据库记录（级联删除 chunks）
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is not None:
            db.delete(doc)

    # 3. 删除磁盘文件
    if remove_file:
        try:
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"已删除文件: {file_path}")
        except OSError as exc:
            logger.warning(f"删除文件失败: {exc}")

    logger.info(f"文档 {document_id} ({stored}) 已删除")
    return True


# ---------- 统计 ----------
def get_stats() -> Dict[str, Any]:
    from app.services.milvus_store import milvus_store

    with session_scope() as db:
        total_docs = db.query(func.count(Document.id)).scalar() or 0
        ready = (
            db.query(func.count(Document.id))
            .filter(Document.status == DocumentStatus.READY)
            .scalar()
            or 0
        )
        failed = (
            db.query(func.count(Document.id))
            .filter(Document.status == DocumentStatus.FAILED)
            .scalar()
            or 0
        )
        processing = (
            db.query(func.count(Document.id))
            .filter(
                Document.status.in_(
                    [
                        DocumentStatus.PENDING,
                        DocumentStatus.PARSING,
                        DocumentStatus.INDEXING,
                    ]
                )
            )
            .scalar()
            or 0
        )
        total_chunks = db.query(func.count(Chunk.id)).scalar() or 0
        total_chars = db.query(func.coalesce(func.sum(Document.char_count), 0)).scalar() or 0

    return {
        "total_documents": total_docs,
        "ready_documents": ready,
        "failed_documents": failed,
        "processing_documents": processing,
        "total_chunks": total_chunks,
        "total_characters": int(total_chars),
        "total_vectors": milvus_store.count(),
    }


def _parse_keywords(raw: Optional[str]) -> List[str]:
    import json

    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [str(x) for x in data] if isinstance(data, list) else []
    except json.JSONDecodeError:
        return [k for k in raw.split(",") if k.strip()]
