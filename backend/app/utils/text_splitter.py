"""
文本分块工具

面向中文优化的递归分块器：
    段落 -> 句子 -> 子句 -> 硬切

不依赖 LangChain 的分块实现，避免不同版本 API 差异带来的问题。
"""
import re
from typing import Dict, List

# 分隔符优先级：从粗到细
_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 换行
    "。",     # 中文句号
    "！",
    "？",
    "；",
    ". ",     # 英文句号
    "! ",
    "? ",
    "; ",
    "，",     # 逗号
    ", ",
    " ",
    "",
]

_WS_RE = re.compile(r"[ \t\u3000]+")


def normalize_text(text: str) -> str:
    """清理空白字符，统一换行"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_recursive(text: str, separators: List[str]) -> List[str]:
    """按分隔符递归切分，保留分隔符本身"""
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return list(text)

    parts: List[str] = []
    if sep not in text:
        return _split_recursive(text, rest)

    # 用 sep 切分并把 sep 贴回前一段，避免句子断裂
    segments = text.split(sep)
    for i, seg in enumerate(segments):
        if not seg:
            continue
        piece = seg + sep if i < len(segments) - 1 else seg
        if len(piece) > 0:
            parts.append(piece)

    # 对仍然过长的片段继续细切
    result: List[str] = []
    for p in parts:
        result.append(p)
    return result


def _merge_pieces(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """把小片段合并成不超过 chunk_size 的块，块间保留 overlap 字符重叠"""
    chunks: List[str] = []
    buffer = ""

    for piece in pieces:
        piece = piece
        if not piece.strip():
            continue

        # 单片超长，硬切
        if len(piece) > chunk_size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            start = 0
            while start < len(piece):
                chunks.append(piece[start : start + chunk_size].strip())
                start += chunk_size - overlap
            continue

        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            # 构造重叠部分
            tail = buffer[-overlap:] if overlap > 0 and buffer else ""
            buffer = tail + piece

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict]:
    """
    将长文本切分为块

    Returns:
        [{"index": 0, "content": "...", "char_count": 123}, ...]
    """
    text = normalize_text(text)
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    # 先按分隔符切成小片段
    pieces = _split_recursive(text, _SEPARATORS)

    # 再合并成目标大小的块
    merged = _merge_pieces(pieces, chunk_size, chunk_overlap)

    return [
        {"index": i, "content": c, "char_count": len(c)}
        for i, c in enumerate(merged)
    ]


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）"""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)
