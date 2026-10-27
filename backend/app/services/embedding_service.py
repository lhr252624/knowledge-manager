"""
嵌入模型服务（支持本地 BGE-M3 / 远程阿里云百炼两种提供方）

设计要点：
- 延迟加载：首次调用时才载入模型，避免拖慢 API 启动
- 单例 + 加锁：全进程只加载一份模型，节省显存
- 归一化：输出向量做 L2 归一化，配合内积(IP)即等价余弦相似度
- 可切换提供方：本地模型与云端 API 走同一套 encode() 接口，上层无感知

两种提供方：
    local     : 本地 sentence-transformers 加载 BGE-M3（默认，离线可用）
    dashscope : 阿里云百炼 OpenAI 兼容 /embeddings 接口（无需 torch、无需模型文件）

为什么远程模式也必须自己做 L2 归一化：
    Milvus 集合用的是 IP（内积）度量，只有向量都是单位长度时内积才等于余弦相似度。
    本地 BGE-M3 由 normalize_embeddings=True 保证，而云端接口返回的是未归一化的原始向量；
    若直接入库，IP 分数会随文本长度漂移，检索阈值全部失效。

关于加载锁（重要）：
    后台索引任务与图谱计算会并发调用 encode()。若不加锁，多个线程会同时
    执行 SentenceTransformer(...)，而 transformers 在加载时先于 meta device
    建模再搬权重，并发实例化会互相踩状态，抛出
        NotImplementedError: Cannot copy out of meta tensor; no data!
    因此必须用双检锁保证"只加载一次、且加载完成前其他线程阻塞等待"。
"""
import os
import re
import threading
import time
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

# 支持 dimensions 参数的模型（百炼 v3/v4 及 qwen3 系列）
_DIM_MODEL_HINTS = ("text-embedding-v3", "text-embedding-v4", "qwen3")


def _provider() -> str:
    return (settings.embedding_provider or "local").strip().lower()


class EmbeddingService:
    """向量化服务（本地 BGE-M3 / 远程百炼，延迟加载单例）"""

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

    # ---------- 提供方 ----------
    @property
    def is_remote(self) -> bool:
        return _provider() == "dashscope"

    @property
    def model_tag(self) -> str:
        """
        向量空间标识：换模型 / 换维度 / 换提供方都会变

        不触发模型加载，可在 import 期安全使用。
        """
        if self.is_remote:
            return f"dashscope:{settings.dashscope_embedding_model}:{settings.embedding_dim}"
        return f"local:{settings.bge_model_path}:{settings.embedding_dim}"

    @property
    def space_tag(self) -> str:
        """向量空间标识的短形式，用于拼 Milvus 集合名"""
        if self.is_remote:
            raw = f"ds_{settings.dashscope_embedding_model}_{settings.embedding_dim}"
        else:
            raw = f"local_{os.path.basename(settings.bge_model_path)}_{settings.embedding_dim}"
        return re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()

    # ---------- 本地模型加载 ----------
    def _load(self) -> None:
        # 远程模式不加载任何本地模型，维度直接取配置
        if self.is_remote:
            self.__class__._model = None
            self.__class__._device = "remote"
            self.__class__._dim = settings.embedding_dim
            return

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

            if self.__class__._dim != settings.embedding_dim:
                logger.warning(
                    f"模型实际维度 {self.__class__._dim} 与配置 embedding_dim="
                    f"{settings.embedding_dim} 不一致，以模型为准"
                )

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
        if self.is_remote:
            return "remote"
        return self._device if self._model is not None else settings.bge_device

    @property
    def provider(self) -> str:
        return _provider()

    def is_loaded(self) -> bool:
        # 远程模式没有"本地模型"，只要配了 Key 就算可用
        if self.is_remote:
            return bool(settings.dashscope_api_key)
        return self._model is not None

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """文本 -> 向量（单条返回 list[float]，多条返回 list[list[float]]）"""
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return []

        if self.is_remote:
            vectors = self._encode_remote(batch, normalize)
        else:
            vectors = self._encode_local(batch, normalize, show_progress)

        return vectors[0] if single else vectors

    def encode_queries(self, queries: Union[str, Sequence[str]]):
        """查询向量化（BGE 系列对查询无需额外指令前缀，保持对称编码）"""
        return self.encode(queries)

    def encode_documents(self, docs: Union[str, Sequence[str]]):
        """文档向量化"""
        return self.encode(docs)

    # ---------- 本地推理 ----------
    def _encode_local(
        self, batch: List[str], normalize: bool, show_progress: bool
    ) -> List[List[float]]:
        self._load()

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
        return vectors.tolist()

    # ---------- 远程推理 ----------
    def _encode_remote(self, batch: List[str], normalize: bool) -> List[List[float]]:
        if not settings.dashscope_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=dashscope 但未配置 DASHSCOPE_API_KEY；"
                "请填写 Key，或把 EMBEDDING_PROVIDER 改回 local"
            )

        # 百炼单次请求最多 10 条（v3/v4），必须自己切片
        step = max(1, min(int(settings.dashscope_batch_size or 10), 10))
        out: List[List[float]] = []

        # 与本地模式共用同一把锁：限制并发请求数，顺带避免触发限流
        with self.__class__._encode_lock:
            for start in range(0, len(batch), step):
                out.extend(self._post_embeddings(batch[start : start + step]))

        return [self._l2(v) for v in out] if normalize else out

    def _post_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用一个批次，返回与 texts 顺序一致的向量列表"""
        import httpx

        url = settings.dashscope_base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": settings.dashscope_embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        # v3/v4 才支持 dimensions，v1/v2 传了会 400
        if any(h in settings.dashscope_embedding_model for h in _DIM_MODEL_HINTS):
            payload["dimensions"] = settings.embedding_dim

        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        attempts = max(0, int(settings.dashscope_max_retries)) + 1
        last_error = ""

        for attempt in range(attempts):
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=settings.dashscope_timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"网络错误: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), len(texts))

                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                # 401/403 是配置问题，重试没有意义
                if resp.status_code in (401, 403):
                    raise RuntimeError(f"百炼鉴权失败，请检查 DASHSCOPE_API_KEY：{last_error}")
                # 只有限流和服务端错误值得重试
                if resp.status_code != 429 and resp.status_code < 500:
                    raise RuntimeError(f"百炼嵌入接口返回错误：{last_error}")

            if attempt < attempts - 1:
                backoff = 1.5 * (2**attempt)
                logger.warning(
                    f"嵌入请求失败（第 {attempt + 1}/{attempts} 次），{backoff:.1f}s 后重试：{last_error}"
                )
                time.sleep(backoff)

        raise RuntimeError(f"百炼嵌入接口连续 {attempts} 次失败：{last_error}")

    def _parse(self, body: dict, expected: int) -> List[List[float]]:
        """解析响应，按 index 还原顺序并校验维度"""
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError(
                f"百炼返回条数异常：期望 {expected}，实际 "
                f"{len(data) if isinstance(data, list) else '非数组'}"
            )

        data = sorted(data, key=lambda item: item.get("index", 0))
        vectors: List[List[float]] = []
        for item in data:
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError("百炼返回了空的 embedding（检查是否误传了 output_type=sparse）")
            if len(vec) != settings.embedding_dim:
                raise RuntimeError(
                    f"向量维度不匹配：期望 {settings.embedding_dim}，实际 {len(vec)}。"
                    f"请把 EMBEDDING_DIM 改成 {len(vec)} 并重建索引"
                )
            vectors.append([float(v) for v in vec])
        return vectors

    @staticmethod
    def _l2(vec: List[float]) -> List[float]:
        """L2 归一化（云端返回的向量不是单位长度，必须自己归一化）"""
        import numpy as np

        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.tolist()
        return (arr / norm).tolist()


# 模块级单例（此时不会加载模型，也不会发请求）
embedding_service = EmbeddingService()
