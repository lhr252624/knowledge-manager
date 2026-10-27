"""
BGE-M3 嵌入模型服务

设计要点：
- 延迟加载：首次调用时才载入模型，避免拖慢 API 启动
- 单例 + 加锁：全进程只加载一份模型，节省显存
- 归一化：输出向量做 L2 归一化，配合内积(IP)即等价余弦相似度

关于加载锁（重要）：
    后台索引任务与图谱计算会并发调用 encode()。若不加锁，多个线程会同时
    执行 SentenceTransformer(...)，而 transformers 在加载时先于 meta device
    建模再搬权重，并发实例化会互相踩状态，抛出
        NotImplementedError: Cannot copy out of meta tensor; no data!
    因此必须用双检锁保证"只加载一次、且加载完成前其他线程阻塞等待"。
"""
import os
import threading
from typing import List, Sequence, Union

from loguru import logger

from app.config import get_settings

# 必须在本模块导入 transformers 之前设置：
# transformers 默认会尝试导入 TensorFlow（本机 conda 环境里装了 TF），
# 首次导入要花 30-40 秒，且我们完全用不到 TF。
# 关掉后 BGE 模型首次加载快一个数量级。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

settings = get_settings()


class EmbeddingService:
    """BGE-M3 向量化服务（延迟加载单例）"""

    _instance: "EmbeddingService | None" = None
    _model = None
    _device: str = "cpu"
    _dim: int = 1024
    _load_lock = threading.Lock()
    _encode_lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- 模型加载 ----------
    def _load(self) -> None:
        # 快路径：已加载则直接返回，不进入锁
        if self.__class__._model is not None:
            return

        with self.__class__._load_lock:
            # 双检：可能在等锁期间已被其他线程加载完成
            if self.__class__._model is not None:
                return

            import torch
            from sentence_transformers import SentenceTransformer

            device = settings.bge_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，回退到 CPU")
                device = "cpu"

            logger.info(f"正在加载 BGE 模型: {settings.bge_model_path} (device={device})")
            model = SentenceTransformer(
                settings.bge_model_path,
                device=device,
                # 关闭 low_cpu_mem_usage 的 meta device 初始化，避免与并发加载
                # 相关的 "Cannot copy out of meta tensor" 问题
                model_kwargs={"low_cpu_mem_usage": False},
            )
            model.max_seq_length = settings.bge_max_length
            model.eval()

            # 先完成全部初始化再发布，避免其他线程看到半成品
            self.__class__._model = model
            self.__class__._device = device
            self.__class__._dim = model.get_sentence_embedding_dimension()

            logger.info(
                f"BGE 模型加载完成 | 设备={device} 维度={self.__class__._dim} "
                f"最大长度={settings.bge_max_length}"
            )

    # ---------- 对外接口 ----------
    @property
    def dimension(self) -> int:
        self._load()
        return self._dim

    @property
    def device(self) -> str:
        return self._device if self._model is not None else settings.bge_device

    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """文本 -> 向量（单条返回 list[float]，多条返回 list[list[float]]）"""
        self._load()

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return []

        # 串行化推理：8GB 显存下并发 encode 容易 OOM，且 CUDA 上批处理并发
        # 带来的收益远小于风险（单批在 GPU 上仅几十毫秒）。
        with self.__class__._encode_lock:
            vectors = self._model.encode(
                batch,
                batch_size=settings.bge_batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        result = vectors.tolist()
        return result[0] if single else result

    def encode_queries(self, queries: Union[str, Sequence[str]]):
        """查询向量化（BGE 系列对查询无需额外指令前缀，保持对称编码）"""
        return self.encode(queries)

    def encode_documents(self, docs: Union[str, Sequence[str]]):
        """文档向量化"""
        return self.encode(docs)


# 模块级单例（此时不会加载模型）
embedding_service = EmbeddingService()
