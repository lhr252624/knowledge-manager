"""
中文分词与文本归一化

问答缓存要知道"两次提问是不是同一个意思"，最简单可靠的办法是：

    归一化 -> jieba 分词 -> 去停用词 -> 词袋

这里只做词袋层面的事，相似度计算在 `app/services/bm25.py`。

注意点：
    - 数字必须保留（"2024 年预算" 和 "2025 年预算" 是两个问题）
    - 单个汉字（"的""了""是"）几乎全是噪声，直接丢；但英文/数字单字保留
    - jieba 首次分词要加载词典（约 0.5s），用惰性初始化 + 锁
"""
import re
import threading
import unicodedata
from functools import lru_cache
from typing import List, Set

import jieba

# jieba 默认会往控制台打日志，问句一多就把日志淹了
jieba.setLogLevel(60)

_init_lock = threading.Lock()
_initialized = False

# 中文停用词（含疑问词/语气词）：它们几乎出现在每个问题里，
# 参与计算只会把不同的问题"拉近"，对区分毫无帮助
STOPWORDS = frozenset(
    """
    的 了 是 在 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 着 没 看 好
    这 那 它 其 与 及 或 并 等 被 把 给 让 从 对 为 以 于 之 而 则 但 却 呢 吗 吧 啊
    一个 一些 一下 一样 一直 一定 一般 这个 那个 这些 那些 这样 那样 这种 那种
    我们 你们 他们 她们 它们 咱们 自己 大家 别人
    什么 怎么 怎样 怎么样 如何 为何 为什么 哪些 哪个 哪里 多少 多久 多大 几个
    请问 请 帮忙 帮我 告诉 一下 能否 是否 可以 可不可以 需要 应该 能 会不会
    以及 还有 或者 还是 如果 但是 因为 所以 然后 不过 并且 而且 其实 只是
    关于 对于 根据 通过 按照 由于 因此 于是 目前 现在 最近 之前 之后 时候
    问 问题 内容 方面 情况 相关 具体 主要 比较 非常 特别 更加 一些 有点
    """.split()
)

# 只保留"有信息量"的字符：中英文、数字
_KEEP_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
_DIGITS_RE = re.compile(r"\d+")

# 否定词：出现与否会改变答案含义
NEGATIONS = frozenset(
    ["不", "没", "没有", "无", "非", "未", "别", "勿", "不能", "不会", "不是", "不要"]
)


def ensure_ready() -> None:
    """确保 jieba 词典已加载（幂等，线程安全）"""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        jieba.initialize()
        _initialized = True


def normalize(text: str) -> str:
    """全角转半角 + 小写 + 去掉标点符号"""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    return _KEEP_RE.sub(" ", s)


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple:
    """
    分词并过滤，返回不可变元组（可缓存、可哈希）

    规则：
        1. 归一化后 jieba 精确模式切分
        2. 丢掉停用词
        3. 中文只保留 ≥2 字的词（单字信息量太低）
        4. 英文/数字保留（长度 ≥1），纯符号丢弃
    """
    cleaned = normalize(text)
    if not cleaned.strip():
        return ()

    ensure_ready()
    out: List[str] = []
    for word in jieba.lcut(cleaned, cut_all=False):
        word = word.strip()
        if not word or word in STOPWORDS:
            continue
        if word.isascii():
            # 英文/数字：单字符也保留（如 "a4"、"2024"）
            if word.isalnum():
                out.append(word)
            continue
        if len(word) < 2:
            continue
        out.append(word)
    return tuple(out)


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def digits(text: str) -> Set[str]:
    """抽取所有数字串（用于"2024 vs 2025"这类必须区分的场景）"""
    return set(_DIGITS_RE.findall(unicodedata.normalize("NFKC", str(text or ""))))


def has_negation(text: str) -> bool:
    """
    是否含否定词

    必须按"词"判断，不能按"字"做子串匹配。旧写法是
        any(neg in normalize(text) for neg in NEGATIONS)
    结果把这两句误判成了否定句：
        "数据库用的是 MySQL 还是别的"  -> 命中了"别"
        "有没有现成的接口文档"        -> 命中了"没"
    它们和各自的改写因此被守卫挡在缓存之外，白白漏掉命中。

    改成 jieba 分词后按词匹配："别的""有没有""无论""差不多"都不再算否定，
    而"不支持"（切成 不 + 支持）、"没有"、"无法"、"不能"、"别删" 仍然能识别。
    """
    return _negation_hit(text)


@lru_cache(maxsize=4096)
def _negation_hit(text: str) -> bool:
    cleaned = normalize(text)
    if not cleaned.strip():
        return False
    ensure_ready()
    return any(word in NEGATIONS for word in jieba.lcut(cleaned, cut_all=False))


def question_key(text: str) -> str:
    """去重用的规范化键（归一化 + 去停用词 + 排序无关）"""
    tokens = tokenize(text)
    return " ".join(sorted(tokens)) if tokens else normalize(text).strip()
