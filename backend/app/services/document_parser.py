"""
文档解析服务

支持格式：
    .pdf  .docx  .doc  .md  .markdown  .txt  .csv  .log  .json
"""
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".md",
    ".markdown",
    ".txt",
    ".csv",
    ".log",
    ".json",
}


def sha256_of_file(path: Path) -> str:
    """流式计算文件哈希（去重用）"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def _read_text_auto(path: Path) -> str:
    """自动识别编码读取纯文本"""
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _parse_pdf(path: Path) -> str:
    """PDF：优先 pdfplumber（版面质量好），失败回退 PyPDF2"""
    texts = []
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                content = page.extract_text() or ""
                if content.strip():
                    texts.append(content)
        if texts:
            return "\n\n".join(texts)
    except Exception as exc:
        logger.warning(f"pdfplumber 解析失败，尝试 PyPDF2: {exc}")

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        for page in reader.pages:
            content = page.extract_text() or ""
            if content.strip():
                texts.append(content)
        return "\n\n".join(texts)
    except Exception as exc:
        raise RuntimeError(f"PDF 解析失败: {exc}") from exc


def _parse_docx(path: Path) -> str:
    """Word 文档"""
    try:
        from docx import Document as Docx

        doc = Docx(str(path))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        # 表格内容
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)
    except Exception as exc:
        raise RuntimeError(f"Word 解析失败: {exc}") from exc


def _parse_csv(path: Path) -> str:
    text = _read_text_auto(path)
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join(r) for r in reader if any(c.strip() for c in r)]
        return "\n".join(rows)
    except Exception:
        return text


def _parse_json(path: Path) -> str:
    text = _read_text_auto(path)
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text


def _guess_title(text: str, filename: str) -> str:
    """从正文首个标题行或文件名推断标题"""
    for line in text.split("\n")[:20]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()[:200]
        if 2 <= len(line) <= 80:
            return line[:200]
    return Path(filename).stem


def parse_document(path: Path) -> Tuple[str, Dict]:
    """
    解析文档

    Returns:
        (正文文本, 元信息字典)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}")

    if ext == ".pdf":
        text = _parse_pdf(path)
    elif ext in (".docx", ".doc"):
        text = _parse_docx(path)
    elif ext == ".csv":
        text = _parse_csv(path)
    elif ext == ".json":
        text = _parse_json(path)
    else:  # md / txt / log
        text = _read_text_auto(path)

    # 清理常见噪声
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("未能从文档中提取到文本内容")

    meta = {
        "title": _guess_title(text, path.name),
        "filename": path.name,
        "file_type": ext.lstrip("."),
        "file_size": path.stat().st_size,
        "char_count": len(text),
        "file_hash": sha256_of_file(path),
    }
    logger.info(f"解析完成 {path.name} | {meta['char_count']} 字符")
    return text, meta
