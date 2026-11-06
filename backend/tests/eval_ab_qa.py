"""
问答 A/B 评测（agent 循环 开 / 关，同一套题、同一批标注）

和 eval_qa_retrieval.py / eval_qa_agent.py 的区别：
    那两个脚本把题目写死在代码里，且题目对应的文档（神经网络报告、调查问卷、
    家庭预算…）早已不在库里，跑出来全是 MISS，没有任何信息量。
    这里把题目和标注拆到 ab_questions.json，题目从**当前库里的 71 篇**里出，
    每条标注一个"必须命中的分块"（document_id + chunk_index + 关键串），
    换库以后只改 JSON，不动代码。

为什么不走 HTTP：
    A/B 要翻 settings.qa_agent_enabled。走 HTTP 就得起两个进程、把 BGE-M3 和
    重排模型各加载一遍（8G 显存容易 OOM），而且改配置还得重启。
    这里直接在进程内调 QAAgent.run()：开关是模块级 settings，翻一下就行，
    模型只加载一次。代价是量不到 SSE 事件，那些由 test_api.py 覆盖。

判据（三档，分开统计，避免"答案对了但资料没到位"被算成成功）：
    答案命中 : 关键串出现在 answer 里            ← 用户能直接看到
    片段命中 : citations 里有 (document_id, chunk_index) 完全一致 ← 资料到位
    文档命中 : citations 里出现了那份文档        ← 最宽松
    refuse 类只有"是否拒答"一项，不参与前两项统计。

说明两点口径：
    B 组不是"每条都跑满两轮"——闸门（证据足够、不需要 agent）照样生效，
        命中的题步数为 0，其余才会进 3~5 步循环，这正是线上真实行为。
    A 组本来就是关掉 agent 的路径，接口会把 degraded 置 True，
        这里不计入"降级（异常）"，只统计 B 组的异常降级。

用法（必须在 backend 目录下，模型和 .env 都按 cwd 找）：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\eval_ab_qa.py                 # A/B 全量
    .\\venv\\Scripts\\python.exe tests\\eval_ab_qa.py --limit 5       # 先跑 5 条试水
    .\\venv\\Scripts\\python.exe tests\\eval_ab_qa.py --only rag_system
    .\\venv\\Scripts\\python.exe tests\\eval_ab_qa.py --configs a     # 只跑 A（不翻开关）
    .\\venv\\Scripts\\python.exe tests\\eval_ab_qa.py --verify-annotations   # 只核对标注，不跑模型

跑一次大约：预热 60~100s（加载 BGE-M3 + 重排），之后 A 每条约 6~12s、
B 每条约 10~35s，15 条合计 10~12 分钟。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

# settings 在 import 时按 cwd 读 .env，所以先切到 backend 再 import
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="ERROR")  # 题目之间刷屏的 INFO 不要，失败才看

from app.agents.qa_agent import qa_agent  # noqa: E402
from app.config import get_settings  # noqa: E402

settings = get_settings()  # 单例；下面翻 qa_agent_enabled 时改的就是 qa_agent 用的那份

QUESTIONS = BACKEND / "tests" / "ab_questions.json"
# 拒答的几种写法：两条固定话术（检索为空 / 相关度低于下限），
# 加上模型自己组织的"知识库里没有…"——这类题问的就是"库里有没有"，答"没有"即为对
REFUSAL_MARKS = (
    "知识库中没有找到",
    "知识库中暂未检索到",
    "知识库里没有",
    "知识库中没有",
    "知识库未提及",
    "未检索到相关",
    "没有找到相关",
)
TOP_K = 5


def load_questions(only: str = "", limit: int = 0):
    if not QUESTIONS.exists():
        print(f"[FAIL] 找不到标注文件 {QUESTIONS}")
        return []
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    items = data.get("questions") or []
    if only:
        items = [q for q in items if q.get("group") == only]
    if limit:
        items = items[:limit]
    return items


def ask(question: str, history=None) -> dict:
    """走和 /qa 相同的内核：QAAgent.run()"""
    started = time.perf_counter()
    result = qa_agent.run(question=question, top_k=TOP_K, history=history or [])
    result["_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _hit(needle: str, text: str) -> bool:
    """关键串是否命中：先逐字比，再忽略空白比一次

    模型写 "P95延迟" 还是 "P95 延迟"、"WeightedRanker(0.7,0.3)" 还是带空格的，
    语义上没区别，不该算答错；但词序、数字仍然严格比。
    """
    if not needle:
        return False
    if needle in text:
        return True
    squeeze = lambda s: "".join(s.split())  # noqa: E731
    return squeeze(needle) in squeeze(text)


def judge(entry: dict, result: dict) -> dict:
    answer = result.get("answer") or ""
    citations = result.get("citations") or []
    key = entry.get("key")
    verdict = {
        "answer_ok": False,
        "chunk_ok": False,
        "doc_ok": False,
        "refused": any(mark in answer for mark in REFUSAL_MARKS),
        "steps": sum(1 for s in (result.get("steps") or []) if isinstance(s.get("step"), int)),
        "ms": result.get("_ms") or 0,
        "degraded": bool(result.get("degraded")),
    }

    if not key:  # refuse 类：拒答就算对
        verdict["answer_ok"] = verdict["refused"]
        return verdict

    needle = key.get("string") or ""
    # string 是"原文里逐字有的"形式（供核对）；also 是同一个答案的常见写法
    # （"项目深度（30%）" vs "30%"）。任一命中就算答对，别让措辞决定指标。
    variants = [needle] + list(key.get("also") or [])
    verdict["answer_ok"] = any(_hit(v, answer) for v in variants if v)
    verdict["hit_on"] = next((v for v in variants if v and _hit(v, answer)), "")
    verdict["doc_ok"] = any(c.get("document_id") == key.get("document_id") for c in citations)
    verdict["chunk_ok"] = any(
        c.get("document_id") == key.get("document_id")
        and c.get("chunk_index") == key.get("chunk_index")
        for c in citations
    )
    return verdict


def eval_one(entry: dict, label: str) -> dict:
    """单条题目；followup 型先问第一问，再带上下文追问，只评追问那一轮"""
    q = entry["question"]
    try:
        first = ask(q)
        if entry.get("type") == "followup" and entry.get("followup"):
            fu = entry["followup"]
            history = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": first.get("answer") or ""},
            ]
            final = ask(fu["question"], history=history)
            verdict = judge({"key": fu.get("key")}, final)
            # 追问要等两轮，耗时按两轮之和算，才和"用户等多久"对得上
            verdict["ms"] = (first.get("_ms") or 0) + (final.get("_ms") or 0)
            verdict["answer"] = final.get("answer") or ""
            verdict["question"] = fu["question"]
        else:
            verdict = judge(entry, first)
            verdict["answer"] = first.get("answer") or ""
            verdict["question"] = q
    except Exception as exc:  # 单题失败不中断整轮
        logger.warning(f"[{label}] {q[:20]} 出错: {exc}")
        verdict = {
            "answer_ok": False,
            "chunk_ok": False,
            "doc_ok": False,
            "refused": False,
            "steps": 0,
            "ms": 0,
            "degraded": False,
            "answer": "",
            "question": q,
            "error": str(exc)[:80],
        }
    if label == "A":
        # A 组本来就是"关掉 agent"的路径，run() 里会把 degraded 置 True，不算异常
        verdict["degraded"] = False
    verdict["id"] = entry.get("id")
    verdict["group"] = entry.get("group")
    verdict["type"] = entry.get("type")
    print(
        f"  [{label}] {verdict['id']:<5} "
        f"{'答对' if verdict['answer_ok'] else '答错':<4} "
        f"{'片' if verdict['chunk_ok'] else '-':<2} "
        f"{'文' if verdict['doc_ok'] else '-':<2} "
        f"{verdict['steps']:>2} 步 {verdict['ms']:>6}ms  {verdict['question'][:34]}"
    )
    return verdict


def summarize(label: str, rows: list) -> dict:
    judged = [r for r in rows if r.get("type") != "refuse"]
    refuse = [r for r in rows if r.get("type") == "refuse"]
    n = len(judged) or 1
    times = sorted(r["ms"] for r in rows) or [0]
    mid = len(times) // 2
    median = times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2
    return {
        "label": label,
        "answer_ok": sum(1 for r in judged if r["answer_ok"]),
        "chunk_ok": sum(1 for r in judged if r["chunk_ok"]),
        "doc_ok": sum(1 for r in judged if r["doc_ok"]),
        "judged": len(judged),
        "refuse_ok": sum(1 for r in refuse if r["refused"]),
        "refuse_total": len(refuse),
        "avg_ms": sum(r["ms"] for r in rows) / (len(rows) or 1),
        "median_ms": median,  # 外部 LLM 偶发几十秒长尾，中位数比均值稳
        "avg_steps": sum(r["steps"] for r in rows) / (len(rows) or 1),
        "degraded": sum(1 for r in rows if r["degraded"]),
        "rows": rows,
    }


def compare(a: dict, b: dict) -> None:
    print("\n对比")
    print(f"  {'指标':<14}{'A（单次）':<16}{'B（agent）':<16}差值")
    for name, key in [
        ("答案命中", "answer_ok"),
        ("片段命中", "chunk_ok"),
        ("文档命中", "doc_ok"),
    ]:
        av, bv, n = a[key], b[key], a["judged"]
        print(f"  {name:<14}{av}/{n} {av / n * 100:>5.0f}%    {bv}/{n} {bv / n * 100:>5.0f}%    {bv - av:+d}")
    if a["refuse_total"]:
        print(f"  {'拒答正确':<14}{a['refuse_ok']}/{a['refuse_total']:<13}{b['refuse_ok']}/{b['refuse_total']}")
    print(f"  {'平均耗时':<14}{a['avg_ms'] / 1000:>6.1f}s        {b['avg_ms'] / 1000:>6.1f}s        {(b['avg_ms'] - a['avg_ms']) / 1000:+.1f}s")
    print(f"  {'耗时中位数':<14}{a['median_ms'] / 1000:>6.1f}s        {b['median_ms'] / 1000:>6.1f}s        {(b['median_ms'] - a['median_ms']) / 1000:+.1f}s")
    print(f"  {'平均循环步数':<14}{a['avg_steps']:>6.1f}         {b['avg_steps']:>6.1f}")
    print(f"  {'降级（异常）':<14}{a['degraded']:>6}         {b['degraded']:>6}")

    # 逐题差异：只列 A/B 结论不同的，给面试讲"哪几条被救回来了"
    diff = [
        (ra, rb)
        for ra, rb in zip(a["rows"], b["rows"])
        if ra.get("type") != "refuse" and ra["answer_ok"] != rb["answer_ok"]
    ]
    if diff:
        print("\n  单题差异")
        for ra, rb in diff:
            direction = "B 救回" if rb["answer_ok"] else "B 变差"
            print(f"    {ra['id']} {direction}: {ra['question'][:40]}")


def verify_annotations(entries: list) -> int:
    """不跑模型，只核对每条 key 能不能在该分块里逐字找到

    标注质量直接决定指标是否可信，所以把这一步做成独立子命令：
    对不上就返回非 0，CI / 提交前都能拦。
    """
    from app.db.database import session_scope
    from app.models.document import Chunk

    ok = 0
    with session_scope() as db:
        for entry in entries:
            key = entry.get("key")
            checks = [("", key)] if key else []
            if entry.get("followup"):
                checks.append(("追问 ", entry["followup"].get("key")))
            if not checks:  # refuse 类没有 key，跳过
                print(f"  [SKIP] {entry['id']} 拒答题，无标注")
                ok += 1
                continue
            for prefix, item in checks:
                if not item:
                    print(f"  [FAIL] {entry['id']} {prefix}缺少 key")
                    continue
                row = (
                    db.query(Chunk)
                    .filter(
                        Chunk.document_id == item.get("document_id"),
                        Chunk.chunk_index == item.get("chunk_index"),
                    )
                    .first()
                )
                if row is None:
                    print(f"  [FAIL] {entry['id']} {prefix}分块不存在: {item}")
                    continue
                if item.get("string") and item["string"] in row.content:
                    print(f"  [OK]   {entry['id']} {prefix}「{item['string']}」")
                    ok += 1
                else:
                    print(f"  [FAIL] {entry['id']} {prefix}「{item.get('string')}」不在分块里")
    print(f"\n  => 通过 {ok} 处标注")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="问答 A/B 评测")
    parser.add_argument("--only", default="", help="只跑某个 group，如 rag_system")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，试水用")
    parser.add_argument("--configs", default="ab", help="ab=两组都跑；a/b=只跑一组")
    parser.add_argument("--out", default="", help="结果 JSON 路径（默认写到仓库根 .tmp/）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（默认先跑一题把模型加载完）")
    parser.add_argument(
        "--verify-annotations",
        action="store_true",
        help="只核对标注：key 是否真的能在指定分块里逐字找到",
    )
    args = parser.parse_args()

    entries = load_questions(args.only, args.limit)
    if args.verify_annotations:
        return verify_annotations(entries)
    if not entries:
        print("[FAIL] 没有题目可跑")
        return 1

    total = len(entries)
    print(f"题目 {total} 条 · top_k={TOP_K} · 模型首次加载可能要 1 分钟，别以为卡死了\n")

    # 预热：第一题会把 BGE-M3 + 重排模型拉进显存（30s 量级），
    # 不预热的话这段开销全记在 A 组第一题头上，平均耗时就没法看了
    if not args.no_warmup:
        t0 = time.perf_counter()
        print("预热中（加载向量/重排模型）…", flush=True)
        try:
            settings.qa_agent_enabled = True
            ask("RAG 是什么？")
            print(f"预热完成 {time.perf_counter() - t0:.0f}s\n")
        except Exception as exc:
            print(f"预热失败（继续跑，首题会慢）：{exc}\n")

    results = {}
    if "a" in args.configs:
        settings.qa_agent_enabled = False
        print("A 组：单次检索直答（qa_agent_enabled=false）")
        results["a"] = summarize("A", [eval_one(e, "A") for e in entries])
        print(f"  => 答案 {results['a']['answer_ok']}/{results['a']['judged']}\n")

    if "b" in args.configs:
        settings.qa_agent_enabled = True
        print("B 组：ReAct 循环（qa_agent_enabled=true）")
        results["b"] = summarize("B", [eval_one(e, "B") for e in entries])
        print(f"  => 答案 {results['b']['answer_ok']}/{results['b']['judged']}")

    if len(results) == 2:
        compare(results["a"], results["b"])

    out = Path(args.out) if args.out else BACKEND.parent / ".tmp" / "ab_eval_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
