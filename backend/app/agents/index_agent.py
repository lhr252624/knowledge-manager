"""
IndexAgent - 文档索引智能体

职责：
    1. 解析文档（PDF / Word / Markdown / TXT ...）
    2. 按文件哈希去重
    3. 写入 MySQL 文档与分块记录
    4. 调用 BGE-M3 生成向量并写入 Milvus
    5. 触发摘要与关键词抽取
"""
import time
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.db.database import session_scope
from app.models.document import Chunk, Document, DocumentStatus
from app.services.document_parser import parse_document
from app.services.milvus_store import milvus_store
from app.utils.text_splitter import split_text

settings = get_settings()


class IndexAgent(BaseAgent):
    """文档索引智能体"""

    name = "index_agent"
    description = "解析文档、切分文本、生成向量并写入知识库"

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        索引单个文档

        Kwargs:
            file_path: 待索引文件路径（必填）
            generate_summary: 是否生成摘要，默认 True
            original_filename: 用户上传时的原始文件名（磁盘上是 UUID，
                               这里用于入库展示，缺省则回退到磁盘文件名）
        """
        raw_path = kwargs.get("file_path")
        if not raw_path:
            raise ValueError("缺少参数 file_path")

        path = Path(raw_path)
        generate_summary = kwargs.get("generate_summary", True)
        original_filename = kwargs.get("original_filename") or path.name
        started = time.time()

        # ---------- 1. 解析 ----------
        text, meta = parse_document(path)

        # 磁盘文件名是 UUID，标题兜底逻辑会拿到无意义的 UUID，
        # 因此这里用原始文件名重新兜底一次
        if meta.get("title") in (None, "", path.stem):
            meta["title"] = Path(original_filename).stem

        # ---------- 2. 去重 ----------
        with session_scope() as db:
            existing = (
                db.query(Document)
                .filter(Document.file_hash == meta["file_hash"])
                .first()
            )
            if existing:
                logger.info(
                    f"文档已存在（hash 相同），跳过: {original_filename} -> id={existing.id}"
                )
                return {
                    "document_id": existing.id,
                    "duplicated": True,
                    "status": existing.status,
                    "chunk_count": existing.chunk_count,
                    "filename": existing.filename,
                }

            # ---------- 3. 建文档记录 ----------
            doc = Document(
                filename=original_filename,
                stored_name=path.name,
                file_path=str(path.resolve()),
                file_type=meta["file_type"],
                file_size=meta["file_size"],
                file_hash=meta["file_hash"],
                title=meta["title"],
                char_count=meta["char_count"],
                status=DocumentStatus.PARSING,
            )
            db.add(doc)
            db.flush()
            document_id = doc.id

        # ---------- 4. 切分并入库 ----------
        try:
            pieces = split_text(
                text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
            if not pieces:
                raise ValueError("分块结果为空")

            with session_scope() as db:
                doc = db.get(Document, document_id)
                assert doc is not None
                doc.status = DocumentStatus.INDEXING
                doc.chunk_count = len(pieces)

                chunk_rows = []
                for piece in pieces:
                    row = Chunk(
                        document_id=document_id,
                        chunk_index=piece["index"],
                        content=piece["content"],
                        char_count=piece["char_count"],
                    )
                    db.add(row)
                    chunk_rows.append(row)
                db.flush()

                payload = [
                    {
                        "chunk_id": row.id,
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                    }
                    for row in chunk_rows
                ]
                doc_id_for_vector = document_id
                filename = doc.filename

            # ---------- 5. 向量化写入 Milvus ----------
            # 文件名与标题一并进入嵌入文本，让"按文档名搜索"也能走向量通道
            pks = milvus_store.insert_chunks(
                document_id=doc_id_for_vector,
                filename=filename,
                chunks=payload,
                title=meta.get("title"),
            )

            with session_scope() as db:
                for row_id, pk in zip([p["chunk_id"] for p in payload], pks):
                    chunk = db.get(Chunk, row_id)
                    if chunk is not None:
                        chunk.milvus_pk = pk

            # ---------- 6. 摘要与关键词 ----------
            # 注意：状态要等摘要写完后才置为 ready，
            # 否则前端轮询会在摘要尚未生成时拿到 ready 且摘要为空。
            summary_info = {}
            if generate_summary:
                try:
                    from app.agents.summary_agent import summary_agent

                    # 只取摘要与关键词，把 agent 的决策轨迹留在库里（summary_trace），
                    # 不往索引结果里塞，保持这里的返回结构与以前一致
                    info = summary_agent.run(
                        text=text, filename=filename, persist=True,
                        document_id=document_id,
                    )
                    summary_info = {
                        "summary": info.get("summary", ""),
                        "keywords": info.get("keywords", []),
                    }
                except Exception as exc:  # 摘要失败不影响文档可检索性
                    logger.warning(f"摘要生成失败（文档仍可检索）: {exc}")

            with session_scope() as db:
                doc = db.get(Document, document_id)
                assert doc is not None
                doc.status = DocumentStatus.READY
                doc.error_msg = None

            elapsed = round(time.time() - started, 2)
            logger.success(
                f"索引完成 id={document_id} 文件={filename} "
                f"分块={len(pieces)} 耗时={elapsed}s"
            )
            return {
                "document_id": document_id,
                "duplicated": False,
                "status": DocumentStatus.READY,
                "filename": filename,
                "title": meta["title"],
                "char_count": meta["char_count"],
                "chunk_count": len(pieces),
                "elapsed_seconds": elapsed,
                **summary_info,
            }

        except Exception as exc:
            logger.exception(f"索引失败: {path.name}")
            with session_scope() as db:
                doc = db.get(Document, document_id)
                if doc is not None:
                    doc.status = DocumentStatus.FAILED
                    doc.error_msg = str(exc)[:2000]
            raise

    def health(self) -> Dict[str, Any]:
        ok_parser = True
        ok_milvus = milvus_store.health()
        return {
            "agent": self.name,
            "parser": ok_parser,
            "milvus": ok_milvus,
            "status": "ok" if ok_milvus else "degraded",
        }


index_agent = IndexAgent()
