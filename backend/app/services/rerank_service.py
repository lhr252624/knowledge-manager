"""
交叉编码器重排服务（bge-reranker-large）

为什么需要它：
    词面向量只认"措辞几乎相同"，语义向量的分数又分不开"前端/后端"这类最小差异
    （两个问题在语义空间里几乎重合）。交叉编码器把两个问题拼在一起过一遍模型，
    能看到"到底是哪个词不一样"，是判定"是不是同一个问题"最有效的手段。

关于加载：
    bge-reranker-large 加载要一分多钟、占 1~2GB 显存，不能放在请求路径上。
    所以启动后由后台线程慢慢加载，加载完成前 `ready` 为 False，
    调用方直接跳过重排（退化成原来的词面匹配），不影响任何请求。

关于输出范围：
    本机模型 predict() 直接返回 0~1 的概率（内部已做 sigmoid）。
    但不同来源的 reranker 权重不一定如此，所以加载时用一对探针文本测一次：
    若输出越界就自己对 logit 做 sigmoid，保证对外永远是 0~1。
"""
import threading
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from app.config import get_settings

settings = get_settings()


class RerankService:
    """交叉编码器单例（延迟 + 后台加载）"""

    _instance: "RerankService | None" = None
    _model = None
    _infer_lock = threading.Lock()
    _load_thread: Optional[threading.Thread] = None
    _needs_sigmoid = False
    state = "idle"  # idle / loading / ready / disabled / failed
    error: Optional[str] = None

    def __new__(cls) -> "RerankService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- 加载 ----------
    @property
    def ready(self) -> bool:
        return self.__class__._model is not None

    @property
    def enabled(self) -> bool:
        return bool(settings.qa_reranker_enabled)

    def warm_async(self) -> None:
        """启动一个后台线程加载模型（幂等）"""
        if not self.enabled:
            self.__class__.state = "disabled"
            return
        if self.ready or self.__class__.state in ("loading", "failed"):
            return
        self.__class__.state = "loading"
        thread = threading.Thread(
            target=self._load, name="reranker-load", daemon=True
        )
        self.__class__._load_thread = thread
        thread.start()

    def _load(self) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"正在加载重排模型: {settings.qa_reranker_path} (device={device})")
            model = CrossEncoder(
                settings.qa_reranker_path,
                device=device,
                max_length=settings.qa_reranker_max_length,
            )

            # 探针：判断输出是概率还是 logit
            probe = model.predict([("同一个问题", "同一个问题")], batch_size=1)
            values = [float(v) for v in probe.reshape(-1)]
            needs_sigmoid = any(v < 0.0 or v > 1.0 for v in values)

            self.__class__._model = model
            self.__class__._needs_sigmoid = needs_sigmoid
            self.__class__.state = "ready"
            logger.info(
                f"重排模型就绪 | 设备={device} 探针输出={values[0]:.3f} "
                f"需要sigmoid={needs_sigmoid}"
            )
        except Exception as exc:
            self.__class__.state = "failed"
            self.__class__.error = str(exc)
            logger.warning(f"重排模型加载失败，问答缓存退化为词面匹配: {exc}")

    # ---------- 推理 ----------
    def score_pairs(
        self, pairs: Sequence[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[float]:
        """给若干 (A, B) 打分，返回 0~1；模型未就绪时返回空列表

        batch_size 不传时一次算完（缓存层只比几条短问句）；
        检索重排的段落长得多，调用方要传一个小批（默认 8），否则显存会顶爆。
        """
        if not pairs or not self.ready:
            return []

        import numpy as np

        batch = max(1, int(batch_size or len(pairs)))
        with self.__class__._infer_lock:
            raw = self._model.predict(list(pairs), batch_size=batch)

        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self.__class__._needs_sigmoid:
            values = 1.0 / (1.0 + np.exp(-values))
        return [float(min(1.0, max(0.0, v))) for v in values]

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.__class__.state,
            "ready": self.ready,
            "path": settings.qa_reranker_path,
            "threshold": settings.qa_rerank_threshold,
            "error": self.__class__.error,
        }


rerank_service = RerankService()
