"""
问答缓存三层命中验证（语义召回 + 重排判定）

覆盖：
    A. 守卫修复：否定词按"词"判断，不再被"别的""有没有"误伤
    B. 三层命中：精确 / 词面 / 重排
    C. 必须拒绝的：最小差异对、数字不同、否定不同
    D. 向量落库与统计
    E. 未命中时回带"最接近的候选"
    F. 命中耗时

重排模型加载要一分多钟，脚本会等；模型缺失时相关用例标记为 SKIP 而不是失败。

用法（不需要后端启动，直接调服务）：
    cd backend
    .\\venv\\Scripts\\python.exe verify_cache_rerank.py
"""

import sys
from pathlib import Path

# 本文件在 backend/tests/ 下，把 backend/ 加进 sys.path 才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.services.qa_cache_service import kb_signature, qa_cache_service  # noqa: E402
from app.services.rerank_service import rerank_service  # noqa: E402
from app.utils.chinese_tokenizer import has_negation  # noqa: E402

settings = get_settings()

passed, failed, skipped = [], [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


def skip(name, reason=""):
    skipped.append(name)
    print(f"  [SKIP] {name} {reason}")


# ======================= A. 守卫 =======================
def test_guards():
    print("\n[A1] 否定词按词判断")
    cases = [
        ("不支持 PDF", True, "真否定"),
        ("支持 PDF", False, "无否定"),
        ("数据库用的是 MySQL 还是别的", False, "「别的」不是否定"),
        ("有没有现成的接口文档", False, "「有没有」不是否定"),
        ("别删这个文档", True, "「别」单独成词仍算否定"),
        ("无论怎么配置", False, "「无论」不是否定"),
        ("差不多就行了", False, "「差不多」不是否定"),
        ("问答缓存不能关闭吗", True, "「不能」"),
    ]
    for text, expect, note in cases:
        check(f"{text} -> {expect}", has_negation(text) is expect, note)


# ======================= 准备数据 =======================
CANON = [
    "知识库支持上传哪些文件格式",
    "这个项目的技术架构是什么",
    "前端怎么启动",
    "问答缓存是怎么命中的",
    "Milvus 用的是什么索引",
    "2024年的家庭预算是多少",
    "支持 PDF 上传吗",
    "系统里有哪些模型",
    "部署这个系统需要哪些依赖",
]


def seed():
    print("\n[准备] 清空缓存并写入 9 条基准问题")
    qa_cache_service.clear()
    for i, q in enumerate(CANON):
        qa_cache_service.store(q, f"答案{i}：{q}", [], signature=kb_signature())
    qa_cache_service.ensure_index(kb_signature(), force=True)
    return qa_cache_service.stats()


def ensure_reranker() -> bool:
    if rerank_service.ready:
        return True
    print("\n[准备] 等待重排模型加载（首次约 60~90 秒）…")
    rerank_service.warm_async()
    deadline = time.time() + 300
    while not rerank_service.ready and time.time() < deadline:
        time.sleep(2)
    return rerank_service.ready


# ======================= B. 三层命中 =======================
def test_tiers():
    print("\n[B1] 精确命中（分词序列一致，标点/虚词不同也算）")
    for query in (
        "知识库支持上传哪些文件格式",
        "知识库支持上传哪些文件格式？",
        "项目的技术架构是怎样的",
    ):
        res = qa_cache_service.lookup_ex(query)
        hit = res["hit"]
        check(
            f"'{query}' 命中",
            hit is not None and res["method"] == "exact",
            f"方式={res['method']}",
        )

    print("\n[B2] 词面通道（BM25 + 加权余弦 >= 0.85）")
    qa_cache_service.store(
        "相似度阈值是怎么设置的", "阈值 0.85", [], signature=kb_signature()
    )
    qa_cache_service.ensure_index(kb_signature(), force=True)
    res = qa_cache_service.lookup_ex("相似度阈值怎么设置")
    check(
        "换语序命中词面通道",
        res["hit"] is not None and res["method"] in ("lexical", "exact"),
        f"方式={res['method']}",
    )

    print("\n[B3] 重排通道（词面完全不同，语义召回 + 交叉编码器判定）")
    if not rerank_service.ready:
        skip("重排通道", "模型未就绪")
        return
    res = qa_cache_service.lookup_ex("Milvus 用的哪种索引类型")
    check(
        "同义改写经重排命中",
        res["hit"] is not None and res["method"] == "rerank",
        f"方式={res['method']} 重排分={(res['near'] or {}).get('rerank')}",
    )
    check(
        "命中的是正确的那条",
        (res["hit"] or {}).get("matched_question") == "Milvus 用的是什么索引",
        str((res["hit"] or {}).get("matched_question")),
    )


# ======================= C. 必须拒绝 =======================
def test_reject():
    print("\n[C1] 守卫拦截")
    for query, why in (
        ("2025年的家庭预算是多少", "数字不同"),
        ("不支持 PDF 上传吗", "肯定/否定不同"),
        ("MySQL 的表名是什么", "主体不同"),
    ):
        res = qa_cache_service.lookup_ex(query)
        check(f"不命中：{query}", res["hit"] is None, why)

    print("\n[C2] 最小差异对：重排必须挡住")
    if not rerank_service.ready:
        skip("最小差异对", "模型未就绪")
        return
    res = qa_cache_service.lookup_ex("后端怎么启动")
    near = res["near"] or {}
    check(
        "「前端/后端」不命中",
        res["hit"] is None,
        f"重排分={near.get('rerank')} 阈值={settings.qa_rerank_threshold}",
    )
    check(
        "但候选被正确找到",
        near.get("matched_question") == "前端怎么启动",
        str(near.get("matched_question")),
    )


# ======================= D. 向量与统计 =======================
def test_storage():
    print("\n[D1] 向量落库")
    stats = qa_cache_service.stats()  # 此时此刻的状态，不是准备阶段快照
    check("每条缓存都有向量", stats["vector_entries"] == stats["entries"], str(stats["entries"]))
    check("语义召回阈值合理", 0.5 <= stats["semantic_recall"] <= 0.95, str(stats["semantic_recall"]))
    check("重排阈值合理", 0.9 <= stats["rerank"]["threshold"] <= 1.0, str(stats["rerank"]["threshold"]))
    check(
        "重排状态可观测",
        stats["rerank"]["state"] in ("idle", "loading", "ready", "failed", "disabled"),
        stats["rerank"]["state"],
    )


# ======================= E. 相近候选 =======================
def test_near():
    print("\n[E1] 未命中时回带最接近的候选")
    res = qa_cache_service.lookup_ex("能上传什么格式的文档")
    near = res["near"] or {}
    check("未命中", res["hit"] is None)
    check(
        "但给出了候选",
        near.get("matched_question") == "知识库支持上传哪些文件格式",
        str(near.get("matched_question")),
    )
    check("候选分数在记录里", near.get("score") is not None, str(near.get("score")))


# ======================= F. 耗时 =======================
def test_latency():
    print("\n[F1] 命中耗时")
    t0 = time.perf_counter()
    for _ in range(50):
        qa_cache_service.lookup_ex("知识库支持上传哪些文件格式")
    exact_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  精确命中：{exact_ms:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(10):
        qa_cache_service.lookup_ex("Milvus 用的哪种索引类型")
    rerank_ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"  重排命中：{rerank_ms:.1f} ms")
    check("精确命中 < 5ms", exact_ms < 5, f"{exact_ms:.2f} ms")
    if rerank_service.ready:
        check("重排命中 < 2000ms", rerank_ms < 2000, f"{rerank_ms:.0f} ms")


def cleanup() -> None:
    """清掉本次写入的合成缓存条目

    seed() 一开始就会整体清空缓存，所以这里再清一次等于把状态还原成"空缓存"，
    不会留下 answer 里写着"答案0：…"这种假条目让用户在界面上看到。
    """
    try:
        removed = qa_cache_service.clear()
        print(f"\n已清理本次写入的合成缓存 {removed} 条")
    except Exception as exc:  # 清理失败不该把测试结论改成失败
        print(f"\n清理合成缓存失败: {exc}")


def main() -> int:
    print("=" * 70)
    print("问答缓存三层命中验证")
    print("=" * 70)

    test_guards()
    seed()
    ensure_reranker()
    test_tiers()
    test_reject()
    test_storage()
    test_near()
    test_latency()
    cleanup()

    print()
    print("=" * 70)
    print(f"通过 {len(passed)} / 跳过 {len(skipped)} / 失败 {len(failed)}")
    if failed:
        for name in failed:
            print(f"  失败：{name}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
