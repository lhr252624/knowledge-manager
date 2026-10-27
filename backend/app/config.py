"""
应用配置管理

配置优先级（后者覆盖前者）：
    1. Settings 类中的默认值
    2. .env.local / .env 环境变量文件
    3. config.override.json（本地覆盖，不进版本库）
    4. 系统环境变量

之所以需要 config.override.json：.env 类文件由安全策略保护，
不支持由自动化工具修改，本地个性化参数（端口、模型路径等）放在
JSON 覆盖文件里更方便，且不会泄露敏感信息。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件位于 <root>/backend/app/config.py）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置类"""

    # ---------- 环境标识 ----------
    environment: str = "development"
    debug: bool = True

    # ---------- API ----------
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    # 允许跨域访问的前端地址。开发时前端走 Vite 代理、生产时走 Nginx 同源代理，
    # 其实都不需要跨域；这里只放行本地调试端口，避免默认对全网开放。
    cors_origins: List[str] = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]

    # ---------- BGE 嵌入模型 ----------
    bge_model_path: str = str(PROJECT_ROOT / "models" / "bge-m3")
    bge_device: str = "cuda"
    bge_batch_size: int = 16
    bge_max_length: int = 1024

    # ---------- DeepSeek LLM ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ---------- MySQL ----------
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_database: str = "knowledge_manager"

    # ---------- Milvus ----------
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "km_chunks"

    # ---------- 文件路径 ----------
    documents_path: str = str(PROJECT_ROOT / "data" / "documents")
    upload_path: str = str(PROJECT_ROOT / "data" / "uploads")

    # ---------- 文本分块 ----------
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------- 检索 ----------
    search_top_k: int = 5
    score_threshold: float = 0.0

    # ---------- 文件名匹配 ----------
    # 文件名/标题参与嵌入（写入向量库前把名字拼进正文）
    embed_with_name: bool = True
    # 名称相似度达到该值即视为"文件名命中"，参与加分与保底召回
    filename_match_threshold: float = 0.45
    # 名称强命中（保底入选）阈值
    filename_strong_threshold: float = 0.6
    # 文件名命中时给向量分加的最大分值（与实际名称相似度相乘）
    filename_boost: float = 0.20
    # 知识图谱里"文件名关联"边的连边阈值
    graph_name_edge_threshold: float = 0.6

    # ---------- 问答缓存 ----------
    # 命中判定：jieba 分词 -> BM25 召回 -> 加权余弦相似度 >= 阈值 直接返回历史答案
    qa_cache_enabled: bool = True
    qa_cache_similarity_threshold: float = 0.85
    # softmax 置信度下限（只在 BM25 召回的 top_k 上计算；0 表示不校验）
    qa_cache_min_confidence: float = 0.25
    qa_cache_top_k: int = 8
    qa_cache_temperature: float = 1.0
    qa_cache_ttl_days: int = 7
    qa_cache_max_entries: int = 2000
    qa_cache_min_question_len: int = 4

    # ---------- 问答缓存：语义召回 + 重排判定 ----------
    # 词面余弦只认"措辞几乎相同"，同义改写（"能上传什么格式的文档"）命中不了。
    # 所以再加一条语义通道负责"捞出候选"，由重排模型负责"判定"。
    # 语义阈值只用于召回，不做最终判定 —— 它分不开"前端/后端"这种最小差异。
    qa_cache_semantic_recall: float = 0.70
    qa_cache_semantic_top_k: int = 5
    # 交叉编码器重排：本地 bge-reranker-large。未配置或加载失败时自动跳过
    qa_reranker_enabled: bool = True
    qa_reranker_path: str = str(PROJECT_ROOT / "models" / "bge-reranker-large")
    # 重排分数（0~1）达到该值才直接返回缓存答案。
    # 0.995 来自 59 对人工标注的标定：零误判，召回率约 42%
    qa_rerank_threshold: float = 0.995
    # 一次最多重排几条候选（每条约 40ms，控制在 3 条以内）
    qa_rerank_candidates: int = 3
    # 重排模型的最大输入长度（问题都很短，128 足够）
    qa_reranker_max_length: int = 128

    # ---------- 历史对话 ----------
    history_max_conversations: int = 200
    history_max_messages: int = 200
    # 追问时带上前几轮对话作为上下文（0 表示不带）
    history_context_rounds: int = 3
    history_context_chars: int = 400

    # env 文件由 _load_env_files() 自行解析（编码自适应），
    # 因此这里不交给 pydantic-settings，避免 UTF-16/GBK 文件报解码错误。
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 派生属性 ----------
    @property
    def mysql_url(self) -> str:
        """SQLAlchemy 同步连接串"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    # ---------- 目录保障 ----------
    def ensure_dirs(self) -> None:
        for p in (self.documents_path, self.upload_path):
            Path(p).mkdir(parents=True, exist_ok=True)


def _decode_bytes(raw: bytes) -> str:
    """
    宽容解码文本

    兼容 Windows 上常见的几种保存方式：
        UTF-8 / UTF-8-BOM / UTF-16LE（PowerShell 5.1 的 Out-File 默认）/ GBK
    """
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _read_env_file(path: Path) -> Dict[str, str]:
    """解析 KEY=VALUE 形式的 env 文件，返回小写键的字典"""
    try:
        text = _decode_bytes(path.read_bytes())
    except OSError:
        return {}

    data: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        # 去掉包裹引号
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            data[key] = value
    return data


def _load_env_files() -> Dict[str, str]:
    """按优先级合并 env 文件（后加载的覆盖先加载的）"""
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env.local",
        PROJECT_ROOT / ".env.local",  # 项目根目录优先级最高
    ]
    merged: Dict[str, str] = {}
    for path in candidates:
        if path.is_file():
            merged.update(_read_env_file(path))
    return merged


def _load_override() -> Dict[str, Any]:
    """
    读取本地覆盖配置 config.override.json

    文件格式为普通 JSON，键名大小写不敏感，例如：
        { "mysql_port": 3306, "bge_model_path": "/opt/models/bge-m3" }
    """
    candidates = [
        PROJECT_ROOT / "config.override.json",
        Path.cwd() / "config.override.json",
    ]
    valid = set(Settings.model_fields.keys())
    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(_decode_bytes(path.read_bytes()))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            return {
                str(k).lower(): v
                for k, v in data.items()
                if str(k).lower() in valid
            }
    return {}


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例

    优先级（后者覆盖前者）：
        类默认值 < env 文件 < config.override.json < 系统环境变量
    """
    valid = set(Settings.model_fields.keys())
    merged: Dict[str, Any] = {}

    merged.update(_load_env_files())
    merged.update(_load_override())

    # 系统环境变量优先级最高
    for key, value in os.environ.items():
        lowered = key.lower()
        if lowered in valid:
            merged[lowered] = value

    return Settings(**merged)


if __name__ == "__main__":
    s = get_settings()
    print("=" * 52)
    print("  配置检查")
    print("=" * 52)
    print(f"  环境        : {s.environment}")
    print(f"  调试模式    : {s.debug}")
    print(f"  BGE 模型    : {s.bge_model_path}")
    print(f"  BGE 设备    : {s.bge_device}")
    print(f"  LLM 模型    : {s.deepseek_model}")
    print(f"  API Key     : {'已配置' if s.deepseek_api_key else '缺失!'}")
    print(f"  MySQL       : {s.mysql_host}:{s.mysql_port}/{s.mysql_database}")
    print(f"  Milvus      : {s.milvus_host}:{s.milvus_port}/{s.milvus_collection}")
    print(f"  文档目录    : {s.documents_path}")
    print("=" * 52)

    missing = []
    if not Path(s.bge_model_path).is_dir():
        missing.append(f"BGE 模型目录不存在: {s.bge_model_path}")
    if not s.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY 未设置")
    if missing:
        print("\n[警告]")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\n[OK] 基础配置完整")
