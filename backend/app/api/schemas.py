"""API 请求/响应模型"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- 检索 ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询内容")
    top_k: int = Field(5, ge=1, le=50)
    document_ids: Optional[List[int]] = None
    group_by_document: bool = True
    use_name_match: bool = Field(True, description="是否启用文件名/标题匹配")


class SearchResponse(BaseModel):
    query: str
    total_chunks: int
    chunks: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    name_matches: List[Dict[str, Any]] = []


# ---------- 问答 ----------
class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(5, ge=1, le=20)
    document_ids: Optional[List[int]] = None
    use_name_match: bool = Field(True, description="是否启用文件名/标题匹配")
    use_cache: bool = Field(True, description="是否使用问答缓存")
    conversation_id: Optional[int] = Field(None, description="会话 id，不传则新建会话")
    save_history: bool = Field(True, description="是否保存到历史对话（脚本调用可关）")


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[Dict[str, Any]] = []
    retrieved: int = 0
    conversation_id: Optional[int] = None
    cached: bool = False
    cache: Optional[Dict[str, Any]] = None
    # 未命中但缓存里有相近问题时，回带候选，便于前端提示"你要问的是不是这个"
    near: Optional[Dict[str, Any]] = None
    # agent 的证据收集轨迹（每次检索/读取），缓存命中时为空
    steps: List[Dict[str, Any]] = []
    queries: List[str] = []
    # 是否因 agent 循环异常退回了单次检索
    degraded: bool = False
    latency_ms: Optional[int] = None


# ---------- 文档 ----------
class DocumentItem(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    file_type: str
    file_size: int
    status: str
    chunk_count: int = 0
    char_count: int = 0
    summary: Optional[str] = None
    error_msg: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    total: int
    page: int
    size: int
    items: List[DocumentItem]


class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[int] = None
    filename: Optional[str] = None
    duplicated: bool = False


class DocumentUpdateRequest(BaseModel):
    """改文件名 / 标题（文件名参与检索，改动会触发向量重算）"""

    filename: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=512)


# ---------- 历史对话 ----------
class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ---------- 图谱 ----------
class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    count: int
    min_score: float


# ---------- 通用 ----------
class MessageResponse(BaseModel):
    success: bool
    message: str
