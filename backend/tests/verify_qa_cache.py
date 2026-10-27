"""
问答缓存验证

覆盖：
    A. 分词 / BM25 / 守卫（纯函数，不需要服务）
    B. 接口层：未命中 -> 命中 -> 语序变体 -> 数字变体 -> 无关问题
    C. 知识库变更后缓存自动失效
    D. 命中 vs 未命中的耗时对比

用法（需要后端已启动）：
    cd backend
    .\\venv\\Scripts\\python.exe verify_qa_cache.py
"""

import sys
from pathlib import Path

# 本文件在 backend/tests/ 下，把 backend/ 加进 sys.path 才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import sys
import time
import uuid

import httpx

BACKEND = "http://127.0.0.1:8000/api/v1"

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


# ======================= A. 算法层 =======================
def test_algorithm():
    from app.services.bm25 import BM25Index, softmax
    from app.utils.chinese_tokenizer import digits, has_negation, normalize, tokenize

    print("\n[A1] 分词")
    tokens = tokenize("这个项目的技术架构是什么")
    check(
        "停用词被过滤",
        "这个" not in tokens and "什么" not in tokens and "项目" in tokens,
        str(list(tokens)),
    )
    check("数字被保留", "2024" in tokenize("2024年的预算是多少"), str(tokenize("2024年的预算是多少")))
    check("单字被丢弃", "的" not in tokenize("的的的预算安排"))
    check("英文保留", "milvus" in tokenize("Milvus 用的是什么索引"))
    check("空输入安全", tokenize("") == () and tokenize("   ！！  ") == ())

    print("\n[A2] 守卫函数")
    check(
        "数字差异可识别",
        digits("2024年的预算") != digits("2025年的预算"),
        f"{digits('2024年的预算')} vs {digits('2025年的预算')}",
    )
    check(
        "否定词可识别",
        has_negation("不支持 PDF") and not has_negation("支持 PDF"),
    )

    print("\n[A3] BM25 + 加权余弦")
    corpus = ["向量数据库用的是什么索引", "下周计划做什么", "家庭预算怎么安排"]
    index = BM25Index([tokenize(q) for q in corpus])
    sim_same = index.similarity(0, tokenize("向量数据库用的是什么索引"))
    sim_other = index.similarity(1, tokenize("向量数据库用的是什么索引"))
    check("同一问题相似度 1.0", abs(sim_same - 1.0) < 1e-6, f"{sim_same:.3f}")
    check("无关问题相似度 0", sim_other == 0.0, f"{sim_other:.3f}")

    probs = softmax([5.0, 1.0, 0.0])
    check("softmax 归一", abs(float(probs.sum()) - 1.0) < 1e-9, f"sum={float(probs.sum()):.6f}")


# ======================= B. 接口层 =======================
def test_api():
    token = uuid.uuid4().hex[:6]
    q1 = f"向量数据库用的是什么索引（缓存测试{token}）"
    q_same = q1
    q_reorder = f"用的是什么索引，向量数据库（缓存测试{token}）"
    q_other = f"下周计划安排了哪些事情（缓存测试{token}）"

    times = {}
    with httpx.Client(timeout=180.0) as c:
        # 预热嵌入模型：首次加载 BGE 要十几秒，会污染冷热耗时对比
        c.post(f"{BACKEND}/search", json={"query": "预热", "top_k": 1})

        print("\n[B1] 首次提问（应未命中，走模型）")
        t0 = time.perf_counter()
        r = c.post(f"{BACKEND}/qa", json={"question": q1, "top_k": 3, "save_history": False})
        times["cold"] = time.perf_counter() - t0
        data = r.json()
        check("接口 200", r.status_code == 200, str(r.status_code))
        check("首次未命中缓存", data.get("cached") is False, f"cached={data.get('cached')}")
        check("有回答", bool(data.get("answer")), f"{len(data.get('answer') or '')} 字")
        first_answer = data.get("answer") or ""

        print("\n[B2] 原样再问（应命中）")
        t0 = time.perf_counter()
        r = c.post(f"{BACKEND}/qa", json={"question": q_same, "top_k": 3, "save_history": False})
        times["hit"] = time.perf_counter() - t0
        data = r.json()
        cache = data.get("cache") or {}
        check("命中缓存", data.get("cached") is True, f"cached={data.get('cached')}")
        check("相似度 1.0", abs(float(cache.get("similarity", 0)) - 1.0) < 1e-6, str(cache))
        check("标记为完全相同", cache.get("exact") is True, str(cache.get("exact")))
        check("回答与首次一致", data.get("answer") == first_answer, "")
        check("引用也一起返回", isinstance(data.get("citations"), list), f"{len(data.get('citations') or [])} 条")
        check(
            "命中明显更快",
            times["hit"] < times["cold"] / 3,
            f"冷 {times['cold']:.2f}s -> 命中 {times['hit']:.3f}s",
        )

        print("\n[B3] 语序变体（应命中，非 exact）")
        r = c.post(f"{BACKEND}/qa", json={"question": q_reorder, "top_k": 3, "save_history": False})
        data = r.json()
        cache = data.get("cache") or {}
        check("命中缓存", data.get("cached") is True, f"cached={data.get('cached')}")
        check(
            "走的是相似度通道",
            cache.get("exact") is False and float(cache.get("similarity", 0)) >= 0.85,
            str(cache),
        )

        print("\n[B4] 无关问题（应未命中）")
        r = c.post(f"{BACKEND}/qa", json={"question": q_other, "top_k": 3, "save_history": False})
        data = r.json()
        check("未命中", data.get("cached") is False, f"cached={data.get('cached')}")

        print("\n[B5] 关闭缓存开关（use_cache=false）")
        r = c.post(
            f"{BACKEND}/qa",
            json={"question": q1, "top_k": 3, "save_history": False, "use_cache": False},
        )
        check("强制不走缓存", r.json().get("cached") is False, f"cached={r.json().get('cached')}")

        print("\n[B6] 缓存统计")
        stats = c.get(f"{BACKEND}/cache/stats").json()
        check("条目数 > 0", stats.get("entries", 0) > 0, f"entries={stats.get('entries')}")
        check("命中次数 > 0", stats.get("total_hits", 0) > 0, f"hits={stats.get('total_hits')}")
        check(
            "统计里有阈值",
            abs(float(stats.get("similarity_threshold", 0)) - 0.85) < 1e-6,
            str(stats.get("similarity_threshold")),
        )
        print(
            f"    本次进程：命中 {stats.get('session_hits')} / 未命中 {stats.get('session_misses')}"
            f"，命中率 {stats.get('session_hit_rate')}"
        )

    return q1, times


def test_guards_direct():
    """数字 / 否定词守卫（直接调服务，避免多花两次模型调用）"""
    from app.services.qa_cache_service import kb_signature, qa_cache_service

    print("\n[B7] 数字与否定词守卫")
    sig = kb_signature()
    qa_cache_service.ensure_index(sig, force=True)

    base_q = f"第 3 章的预算是多少 {uuid.uuid4().hex[:6]}"
    qa_cache_service.store(base_q, "第三章预算为 12000 元。", [], signature=sig)

    hit = qa_cache_service.lookup(base_q, sig)
    check("基准问题可命中", hit is not None, str(hit and hit["similarity"]))

    hit_other_digit = qa_cache_service.lookup(base_q.replace("第 3 章", "第 5 章"), sig)
    check("数字不同不命中", hit_other_digit is None, "")

    neg_q = f"支持 PDF 上传吗 {uuid.uuid4().hex[:6]}"
    qa_cache_service.store(neg_q, "支持 PDF 上传。", [], signature=sig)
    hit_neg = qa_cache_service.lookup(neg_q.replace("支持", "不支持"), sig)
    check("否定词不同不命中", hit_neg is None, "")


# ======================= C. 失效 =======================
def test_invalidation(c: httpx.Client, question: str):
    from app.services.qa_cache_service import kb_signature

    print("\n[C] 知识库变更后缓存失效")
    sig_before = kb_signature()

    r = c.post(f"{BACKEND}/qa", json={"question": question, "top_k": 3, "save_history": False})
    check("变更前命中", r.json().get("cached") is True, f"cached={r.json().get('cached')}")

    token = uuid.uuid4().hex[:8]
    doc = f"# 缓存失效测试 {token}\n\n" + ("这段文字用于触发知识库指纹变化。\n" * 30)
    files = {"file": (f"缓存失效测试-{token}.md", io.BytesIO(doc.encode("utf-8")), "text/markdown")}
    r = c.post(f"{BACKEND}/documents/upload", files=files, params={"generate_summary": False})
    check("上传测试文档", r.status_code == 200, r.json().get("message", ""))

    # 等索引完成（文档状态变化也会让指纹变化）
    doc_id = None
    deadline = time.time() + 120
    while time.time() < deadline:
        items = c.get(f"{BACKEND}/documents", params={"size": 100}).json().get("items", [])
        target = next((i for i in items if i["filename"] == f"缓存失效测试-{token}.md"), None)
        if target and target["status"] in ("ready", "failed"):
            doc_id = target["id"]
            break
        time.sleep(1.5)
    check("索引完成", doc_id is not None, f"doc_id={doc_id}")

    sig_after = kb_signature()
    check("知识库指纹变化", sig_before != sig_after, f"{sig_before} -> {sig_after}")

    r = c.post(f"{BACKEND}/qa", json={"question": question, "top_k": 3, "save_history": False})
    check(
        "变更后同一问题不再命中",
        r.json().get("cached") is False,
        f"cached={r.json().get('cached')}",
    )

    if doc_id:
        c.delete(f"{BACKEND}/documents/{doc_id}")
    return sig_after


# ======================= 主流程 =======================
def main() -> int:
    try:
        test_algorithm()
        question, times = test_api()
        with httpx.Client(timeout=180.0) as c:
            test_guards_direct()
            test_invalidation(c, question)

            print("\n[D] 清理：清空本次产生的缓存")
            r = c.delete(f"{BACKEND}/cache")
            check("清空缓存接口", r.status_code == 200, r.json().get("message", ""))
            stats = c.get(f"{BACKEND}/cache/stats").json()
            check("清空后条目为 0", stats.get("entries") == 0, f"entries={stats.get('entries')}")

            print("\n[E] 耗时对比")
            print(f"    未命中（走 BGE + Milvus + DeepSeek）: {times['cold']:.2f}s")
            print(f"    命中（分词 + BM25 + 余弦）: {times['hit'] * 1000:.0f}ms")
            if times["hit"]:
                print(f"    加速比: {times['cold'] / times['hit']:.0f}x")
    except Exception as exc:  # noqa: BLE001
        print(f"\n测试异常: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 56)
    print(f"  通过 {len(passed)} 项 / 失败 {len(failed)} 项")
    if failed:
        print("  失败项:", ", ".join(failed))
    print("=" * 56)
    return 0 if not failed else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
