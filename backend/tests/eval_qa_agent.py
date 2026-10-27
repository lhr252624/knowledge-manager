"""
问答质量评测（agent 开 / 关 用同一套指标对比）

衡量的是用户能感知的东西：**答案里有没有正确信息**，以及等了多少秒。
判据：关键串出现在答案里，或出现在引用的 preview 里（后者覆盖"答得简略但资料对"）。

为什么用 /qa 而不是 /search：
    改造成 agent 之后，检索由模型自己发起、可能多次。用 /search 只能量到"单次固定
    检索"的效果，量不到 agent 的行为。这里直接看最终答案，agent 开关一拨就能做对照。

两组用例：
    单轮  : 普通提问
    多轮  : 先问一题，再追问代词句，看上下文有没有真正参与

用法：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\eval_qa_agent.py            # 当前配置
    .\\venv\\Scripts\\python.exe tests\\eval_qa_agent.py 改造前     # 只影响输出标题
"""
import sys

import httpx

API = "http://127.0.0.1:8000/api/v1"

SINGLE_TURN = [
    ("这份报告最终测试准确率是多少？", "89.66"),
    ("实验里学习率设成了多少？", "0.001"),
    ("Dropout 用了多大的比例？", "0.3"),
    ("比较的损失函数是哪一个？", "交叉熵"),
    ("这份报告比较了哪几种激活函数？", "Tanh"),
    ("问卷一共发放了多少份？", "113"),
    ("家庭月结余大约多少？", "1.2"),
    ("建议保留多少备用金？", "12"),
    ("资产配置的比例是多少？", "4:4:2"),
    ("高频缓存的拦截率目标是多少？", "80"),
    ("系统的 QPS 目标是多少？", "1000"),
    ("BM25 相似度阈值判定为多少？", "0.85"),
]

MULTI_TURN = [
    ("这份报告比较了哪些激活函数和优化器？", "那第二点呢", "0.001"),
    ("家庭预算里有哪些配置建议？", "那资产配置呢", "4:4:2"),
    ("教育方案分了哪几个阶段？", "第二阶段做了什么", "0.85"),
]


def ask(question, conversation_id=None):
    payload = {
        "question": question,
        "use_cache": False,
        "save_history": True,
        "top_k": 5,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = httpx.post(f"{API}/qa", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def grounded(data, needle):
    """关键串是否出现在答案或引用预览里"""
    if needle in (data.get("answer") or ""):
        return True
    for c in data.get("citations") or []:
        if needle in (c.get("preview") or ""):
            return True
    return False


def run_single():
    rows = []
    for question, needle in SINGLE_TURN:
        try:
            data = ask(question)
            ok = grounded(data, needle)
        except Exception as exc:
            rows.append((question, False, 0, 0, str(exc)[:40]))
            continue
        rows.append(
            (question, ok, len(data.get("steps") or []), data.get("latency_ms") or 0, "")
        )
    return rows


def run_multi():
    rows = []
    for first, follow, needle in MULTI_TURN:
        conv = None
        try:
            r1 = ask(first)
            conv = r1.get("conversation_id")
            data = ask(follow, conversation_id=conv)
            ok = grounded(data, needle)
        except Exception as exc:
            rows.append((f"{first} → {follow}", False, 0, 0, str(exc)[:40]))
            continue
        finally:
            if conv:
                try:
                    httpx.delete(f"{API}/conversations/{conv}", timeout=30)
                except Exception:
                    pass
        rows.append(
            (
                f"{first} → {follow}",
                ok,
                len(data.get("steps") or []),
                data.get("latency_ms") or 0,
                "",
            )
        )
    return rows


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'结果':<7} {'步数':<5} {'耗时':<8} 问题")
    hit = 0
    total_ms = 0
    for question, ok, steps, ms, err in rows:
        if ok:
            hit += 1
        total_ms += ms
        flag = "[OK]  " if ok else "[MISS]"
        print(f"  {flag} {steps:<5} {ms:<8} {question}" + (f"  {err}" if err else ""))
    n = len(rows)
    print(f"  => 正确 {hit}/{n}（{hit / n * 100:.0f}%） 平均耗时 {total_ms / n / 1000:.1f}s")
    return hit, n, total_ms / n


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "当前配置"
    print("=" * 60)
    print(f"  问答评测 · {label}")
    print("=" * 60)

    s_hit, s_n, s_ms = report("单轮", run_single())
    m_hit, m_n, m_ms = report("多轮追问", run_multi())

    print("\n汇总")
    print(f"  单轮     : {s_hit}/{s_n} = {s_hit / s_n * 100:.0f}%   平均 {s_ms / 1000:.1f}s")
    print(f"  多轮追问 : {m_hit}/{m_n} = {m_hit / m_n * 100:.0f}%   平均 {m_ms / 1000:.1f}s")
    print(
        f"  合计     : {s_hit + m_hit}/{s_n + m_n} = "
        f"{(s_hit + m_hit) / (s_n + m_n) * 100:.0f}%   平均 {(s_ms + m_ms) / 2000:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
