"""
文件名 / 标题匹配工具

背景：知识库里的文档既有「文件名」（用户上传时的原始名字，如
"作业2-王小明.pdf"），也有「标题」（从正文里抽出来的第一行/文档名）。
两者都是用户描述一篇文档时最自然的说法，但纯向量检索几乎感知不到它们
—— 用户问"周报里写了什么"时，向量只看到了正文，"周报"这两个字在文件名里。

这里提供一套不依赖分词库的中文友好匹配：

    1. normalize()  归一化：去扩展名、去分隔符、全角转半角、转小写
    2. NameKey      把文件名/标题切成「片段」（按 -_ 空格 等分隔符）
    3. similarity() 计算一个提问与一个名字的匹配度（0~1）

匹配是分层递进的，规则从强到弱：
    完全相等 1.00 > 所有片段都出现 1.00 > 名字整体出现在提问里 0.98
    > 提问恰好等于某个片段 0.94 > 长片段出现 0.85/0.90
    > 短片段出现 0.70（需过滤"项目""文档"这类泛词）
    > 有辨识度的长公共子串 0.69~0.85 > 二元组重叠兜底
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

# 磁盘上的 UUID 占位文件名（历史遗留：上传时没保存原始文件名）
_UUID_RE = re.compile(r"^[0-9a-f]{32}\.[A-Za-z0-9]{1,8}$", re.I)

# 扩展名
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")

# 片段分隔符（中英文标点 + 空白）
_SEPARATORS = r"\s_\-\u3000·•,.;:!?()\[\]{}<>/\\|'\"`~@#$%^&*+=，。；：！？（）【】《》、…—“”‘’\""
_SEP_RE = re.compile("[" + re.escape(_SEPARATORS) + "]+")

# 太短的片段没有区分度
MIN_SEGMENT_LEN = 2
# 提问里出现片段时的最短长度（3 字以上才单独算强信号）
MIN_SEGMENT_IN_QUERY = 3

# 文件名里常见的泛词：单独命中没有区分度，必须配合其它片段
GENERIC_TOKENS = frozenset(
    {
        "项目", "文档", "文件", "报告", "作业", "论文", "方案", "设计", "系统",
        "计划", "总结", "记录", "资料", "笔记", "模板", "草稿", "副本", "复件",
        "复制", "测试", "分析", "研究", "第", "版", "新", "旧", "终", "初",
        "final", "copy", "new", "old", "test", "draft", "doc", "file", "report",
    }
)


def is_placeholder_name(name: Optional[str]) -> bool:
    """判断是否为历史遗留的 UUID 占位文件名"""
    return bool(_UUID_RE.match((name or "").strip()))


def strip_ext(name: str) -> str:
    """去掉扩展名"""
    return _EXT_RE.sub("", (name or "").strip())


def normalize(text: Optional[str]) -> str:
    """归一化：NFKC + 小写 + 去扩展名 + 去分隔符"""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = strip_ext(s)
    return _SEP_RE.sub("", s)


def _segments(text: Optional[str]) -> List[str]:
    """按分隔符切片，返回归一化后的片段（保序、去重）"""
    if not text:
        return []
    raw = unicodedata.normalize("NFKC", strip_ext(str(text))).lower()
    seen = set()
    out: List[str] = []
    for piece in _SEP_RE.split(raw):
        piece = piece.strip()
        if len(piece) < MIN_SEGMENT_LEN or piece in seen:
            continue
        # 纯年份/纯序号这种片段没有区分度
        if piece.isdigit():
            continue
        seen.add(piece)
        out.append(piece)
    return out


class NameKey:
    """一个文件名或标题的检索用表示"""

    __slots__ = ("raw", "norm", "segments", "bigrams")

    def __init__(self, raw: Optional[str]):
        self.raw = raw or ""
        stem = strip_ext(self.raw)
        self.norm = normalize(stem)
        self.segments = _segments(stem)
        self.bigrams = _bigrams(self.norm)

    def __bool__(self) -> bool:
        return bool(self.norm)

    # 归一化后相同即等价，便于做 lru_cache 的键
    def __hash__(self) -> int:
        return hash(self.norm)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NameKey) and self.norm == other.norm

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"NameKey({self.raw!r})"


def _bigrams(text: str) -> set:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _bigram_score(query: NameKey, name: NameKey) -> float:
    """
    二元组重叠度

    cover     —— 名字里有多少比例的字二元组被提问覆盖（主指标）
    precision —— 提问里的二元组有多少落在名字上（防止长提问堆出高分）
    """
    if not query.bigrams or not name.bigrams:
        return 0.0
    inter = len(query.bigrams & name.bigrams)
    if not inter:
        return 0.0
    cover = inter / len(name.bigrams)
    precision = inter / len(query.bigrams)
    return round(cover * (0.5 + 0.5 * precision), 3)


def _longest_common_substring(a: str, b: str) -> int:
    """最长公共子串长度（difflib 底层是 C 实现，短字符串足够快）"""
    if not a or not b:
        return 0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size


def similarity(query: NameKey, name: NameKey) -> float:
    """
    提问与名字的匹配度（0~1）

    query 一般是用户输入（可能是整句提问），name 是文件名或标题。
    """
    qn, nn = query.norm, name.norm
    if not qn or not nn:
        return 0.0

    # 1. 完全一致
    if qn == nn:
        return 1.0

    # 2. 名字由多段组成且每一段都出现在提问里（"作业2-王小明" ← "王小明的作业2"）
    segs = name.segments
    if len(segs) >= 2 and all(s in qn for s in segs):
        return 1.0

    # 3. 整个名字出现在提问里（"周报-知识库项目进展" ← "周报-知识库项目进展讲了什么"）
    if len(nn) >= MIN_SEGMENT_IN_QUERY and nn in qn:
        return 0.98

    # 4. 提问恰好就是名字里的某一段（"周报" ← "周报-知识库项目进展"）
    best = 0.0
    for seg in segs:
        if seg == qn:
            best = max(best, 0.94)
        elif len(seg) >= MIN_SEGMENT_IN_QUERY and seg in qn:
            # 长片段被提问包含，长度越长越可信
            best = max(best, 0.90 if len(seg) >= 4 else 0.85)
        elif (
            len(seg) == 2
            and seg in qn
            and seg not in GENERIC_TOKENS
        ):
            # 两字片段（如"周报"）只在非泛词时算弱命中
            best = max(best, 0.70)

    # 5. 提问被名字包含（"知识库" ← "知识库项目进展"）
    if len(qn) >= MIN_SEGMENT_IN_QUERY and qn in nn:
        best = max(best, 0.85)

    if best:
        return round(best, 3)

    # 6. 提问与名字共享一段"有辨识度的长片段"（"手写数字识别" ←
    #    "基于人工神经网络的手写数字识别技术报告"），要求片段至少 4 字
    #    且覆盖较短一方 60% 以上，避免"项目""报告"这类泛词乱命中
    lcs = _longest_common_substring(qn, nn)
    shortest = min(len(qn), len(nn))
    if lcs >= 4 and shortest and lcs / shortest >= 0.6:
        return round(min(0.85, 0.45 + 0.06 * lcs), 3)

    # 7. 兜底：字二元组重叠
    return _bigram_score(query, name)


def match_score(query: NameKey, filename: Optional[str], title: Optional[str]) -> Tuple[float, str]:
    """
    同时比对文件名与标题，取更高者

    Returns:
        (score, 命中的字段名 "filename" / "title" / "")
    """
    if not query:
        return 0.0, ""

    file_score = similarity(query, NameKey(filename))
    title_score = 0.0
    if title and normalize(title) and normalize(title) != normalize(filename):
        # 标题是从正文里抽的，可靠性略低于用户亲自命名的文件名
        title_score = round(similarity(query, NameKey(title)) * 0.95, 3)

    if file_score <= 0 and title_score <= 0:
        return 0.0, ""
    if file_score >= title_score:
        return file_score, "filename"
    return title_score, "title"


def similarity_between(a: Optional[str], b: Optional[str]) -> float:
    """
    两个名字之间的相似度（图谱里判断"同系列文档"用）

    `similarity()` 本身是有方向的（它把第二个参数当"名字"、第一个当"提问"，
    规则 2/3/5 都依赖谁是名字），所以这里两个方向都算一遍取较大值，
    保证"文档 A 和文档 B"跟"B 和 A"得到同一个分数。
    """
    ka, kb = NameKey(a), NameKey(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    return max(similarity(ka, kb), similarity(kb, ka))


def build_embed_text(
    filename: Optional[str], title: Optional[str], content: str
) -> str:
    """
    构造写入向量库前的嵌入文本

    把文件名与标题拼在正文前面，让"搜文件名"也能走进向量通道：
    用户问"作业2 王小明 的结论"，向量空间里该文档的每个分块都带着
    "作业2-王小明"这段上下文，比只编码正文更容易被召回。
    """
    stem = strip_ext(filename or "").strip()
    parts: List[str] = []
    if stem and not is_placeholder_name(filename or ""):
        parts.append(f"《{stem}》")
    title = (title or "").strip()
    if title and title != stem:
        parts.append(title)
    parts.append(content or "")
    return "\n".join(p for p in parts if p)
