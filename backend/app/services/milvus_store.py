"""
Milvus 向量存储服务

集合结构：
    pk          INT64    主键（auto_id）
    document_id INT64    关联 MySQL documents.id
    chunk_id    INT64    关联 MySQL chunks.id
    chunk_index INT64    块序号
    content     VARCHAR  块文本（便于直接返回，省一次查库）
    filename    VARCHAR  来源文件名
    embedding   FLOAT_VECTOR(1024)
"""
import threading
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.config import get_settings
from app.services.embedding_service import embedding_service
from app.utils.name_matcher import build_embed_text

settings = get_settings()

_ALIAS = "km_default"


class MilvusStore:
    """Milvus 封装（延迟连接 + 线程安全单例）"""

    _instance: "MilvusStore | None" = None
    _lock = threading.Lock()
    _collection: Optional[Collection] = None
    _ready = False

    def __new__(cls) -> "MilvusStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- 初始化 ----------
    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._connect()
            self._ensure_collection()
            self._ready = True

    def _connect(self) -> None:
        logger.info(f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")
        connections.connect(
            alias=_ALIAS,
            host=settings.milvus_host,
            port=str(settings.milvus_port),
            timeout=10,
        )
        logger.info("Milvus 连接成功")

    def _build_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_service.dimension,
            ),
        ]
        return CollectionSchema(fields=fields, description="知识库文档分块向量")

    def _ensure_collection(self) -> None:
        name = settings.milvus_collection
        if utility.has_collection(name, using=_ALIAS):
            self._collection = Collection(name, using=_ALIAS)
            logger.info(
                f"已存在集合 {name}，当前实体数 {self._collection.num_entities}"
            )
        else:
            logger.info(f"创建集合 {name} ...")
            self._collection = Collection(
                name=name, schema=self._build_schema(), using=_ALIAS
            )
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            self._collection.create_index(
                field_name="embedding", index_params=index_params
            )
            logger.info(f"集合 {name} 创建完成，索引类型 HNSW/IP")

        self._collection.load()

    @property
    def collection(self) -> Collection:
        self._ensure_ready()
        assert self._collection is not None
        return self._collection

    # ---------- 写入 ----------
    def insert_chunks(
        self,
        document_id: int,
        filename: str,
        chunks: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> List[int]:
        """
        写入分块向量

        Args:
            document_id: MySQL 文档 ID
            filename: 文件名
            chunks: [{"chunk_id": int, "chunk_index": int, "content": str}, ...]
            title: 文档标题（与文件名一起并入嵌入文本，提升按名字检索的命中率）

        Returns:
            Milvus 主键列表，顺序与传入一致
        """
        if not chunks:
            return []

        contents = [c["content"] for c in chunks]
        # 落库的 content 保持原文；参与编码的文本额外带上文件名与标题
        texts = [
            build_embed_text(filename, title, text) if settings.embed_with_name else text
            for text in contents
        ]
        vectors = embedding_service.encode_documents(texts)
        assert isinstance(vectors, list)

        entities = [
            [document_id] * len(chunks),
            [int(c["chunk_id"]) for c in chunks],
            [int(c["chunk_index"]) for c in chunks],
            contents,
            [filename] * len(chunks),
            vectors,
        ]
        result = self.collection.insert(entities)
        self.collection.flush()

        pks = [int(pk) for pk in result.primary_keys]
        logger.info(f"文档 {document_id} 写入 {len(pks)} 条向量")
        return pks

    # ---------- 查询 ----------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[Iterable[int]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        top_k = top_k or settings.search_top_k
        floor = settings.score_threshold if min_score is None else min_score
        vector = embedding_service.encode_queries(query)
        assert isinstance(vector, list)

        expr = None
        if document_ids:
            ids = [str(int(i)) for i in document_ids]
            expr = f"document_id in [{', '.join(ids)}]"

        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": max(64, top_k * 8)}},
            limit=top_k,
            expr=expr,
            output_fields=["document_id", "chunk_id", "chunk_index", "content", "filename"],
        )

        hits: List[Dict[str, Any]] = []
        for hit_list in results:
            for hit in hit_list:
                score = float(hit.score)
                if score < floor:
                    continue
                entity = hit.entity
                hits.append(
                    {
                        "milvus_pk": int(hit.id),
                        "score": round(score, 4),
                        "document_id": int(entity.get("document_id")),
                        "chunk_id": int(entity.get("chunk_id")),
                        "chunk_index": int(entity.get("chunk_index")),
                        "content": entity.get("content"),
                        "filename": entity.get("filename"),
                    }
                )
        return hits

    # ---------- 删除 ----------
    def delete_by_document(self, document_id: int) -> int:
        """删除某文档的全部向量"""
        expr = f"document_id == {int(document_id)}"
        result = self.collection.delete(expr)
        self.collection.flush()
        count = result.delete_count if hasattr(result, "delete_count") else 0
        logger.info(f"删除文档 {document_id} 的向量，共 {count} 条")
        return int(count)

    # ---------- 统计 ----------
    def count(self) -> int:
        """
        统计有效向量数

        不能用 collection.num_entities：Milvus 的删除是标记删除，
        该属性在 compaction 之前不会减少，会把已删除的文档算进去。
        改用 count(*) 聚合查询，结果已排除被删除的行。
        """
        try:
            rows = self.collection.query(expr="pk >= 0", output_fields=["count(*)"])
            if rows:
                return int(rows[0].get("count(*)", 0))
            return 0
        except Exception as exc:
            logger.warning(f"count(*) 统计失败，回退 num_entities: {exc}")
            try:
                return int(self.collection.num_entities)
            except Exception:
                return 0

    def drop_all(self) -> None:
        """清空集合（开发调试用）"""
        name = settings.milvus_collection
        if utility.has_collection(name, using=_ALIAS):
            utility.drop_collection(name, using=_ALIAS)
            self._ready = False
            self._collection = None
            logger.warning(f"已删除集合 {name}")

    def health(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception as exc:
            logger.error(f"Milvus 健康检查失败: {exc}")
            return False


milvus_store = MilvusStore()
