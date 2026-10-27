"""
检索质量评测（改 agent 之前先量基线）

为什么需要它：
    项目原有测试覆盖的是"功能是否通"（缓存命中、文件名通道、SSE 事件），
    没有一项衡量"该召回的东西有没有召回"。所以在把 QA 改成 agent 之前，
    先用一组固定的问题把基线钉死，否则改完只能说"感觉好了"。

判据（刻意只做检索级，不做答案级）：
    每个用例给一个"必须出现在召回片段里"的关键串。答案对不对受提示词影响，
    而这里要衡量的是"模型拿到的资料对不对"——这是 agent 改造真正能动的地方。

两组用例：
    单轮  : 普通提问，衡量基础召回
    多轮  : 先问一题，再追问"那第二点呢"，衡量对话上下文有没有参与检索
            （当前实现只用追问原句检索，这组必然大面积失败，那就是基线）

不硬编码文档名：用户文件名里含个人信息，测试里只按内容关键串匹配。

用法（需要后端已启动）：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\eval_qa_retrieval.py
"""
import sys

import httpx

API = "http://127.0.0.1:8000/api/v1"

# (问题, 召回的片段里必须出现的关键串, 说明)
SINGLE_TURN = [
    ("这份报告最终测试准确率是多少？", "89.66", "神经网络报告 · 结论数字"),
    ("实验里学习率设成了多少？", "0.001", "神经网络报告 · 超参数"),
    ("Dropout 用了多大的比例？", "0.3", "神经网络报告 · 正则化"),
    ("比较的损失函数是哪一个？", "交叉熵", "神经网络报告 · 损失函数"),
    ("这份报告比较了哪几种激活函数？", "Tanh", "神经网络报告 · 激活函数"),
    ("问卷一共发放了多少份？", "113", "调查问卷 · 样本量"),
    ("家庭月结余大约多少？", "1.2", "预算文档 · 收支"),
    ("建议保留多少备用金？", "12", "预算文档 · 备用金"),
    ("资产配置的比例是多少？", "4:4:2", "预算文档 · 配置比例"),
    ("高频缓存的拦截率目标是多少？", "80", "教育方案 · 指标"),
    ("系统的 QPS 目标是多少？", "1000", "教育方案 · 指标"),
    ("BM25 相似度阈值判定为多少？", "0.85", "教育方案 · 阈值"),
]

# (第一轮问题, 追问, 追问时召回里必须出现的关键串, 说明)
MULTI_TURN = [
    ("这份报告比较了哪些激活函数和优化器？", "那第二点呢", "Adam", "追问指向「优化器」那一项"),
    ("家庭预算里有哪些配置建议？", "那资产配置呢", "4:4:2", "追问指向配置比例"),
    ("教育方案分了哪几个阶段？", "第二阶段做了什么", "0.85", "追问指向第二阶段细节"),
]


def search(query, top_k=5):
    r = httpx.post(f"{API}/search", json={"query": query, "top_k": top_k}, timeout=120)
    r.raise_for_status()
    return r.json()


def find(data, needle):
    """在召回片段的完整正文里找关键串，返回排名（0 表示没找到）与 top1 分数"""
    chunks = data.get("chunks") or []
    for rank, c in enumerate(chunks, start=1):
        if needle in (c.get("content") or ""):
            return rank, chunks[0].get("score", 0) if chunks else 0
    return 0, chunks[0].get("score", 0) if chunks else 0


def run_single():
    rows = []
    for question, needle, note in SINGLE_TURN:
        try:
            rank, top1 = find(search(question), needle)
        except Exception as exc:  # 单个用例失败不该中断整轮
            rows.append((question, needle, note, 0, 0, str(exc)[:40]))
            continue
        rows.append((question, needle, note, rank, top1, ""))
    return rows


def run_multi():
    """多轮：按当前实现，检索用的就是追问原句，所以这里直接用它检索"""
    rows = []
    for first, follow, needle, note in MULTI_TURN:
        try:
            rank, top1 = find(search(follow), needle)
        except Exception as exc:
            rows.append((f"{first} → {follow}", needle, note, 0, 0, str(exc)[:40]))
            continue
        rows.append((f"{first} → {follow}", needle, note, rank, top1, ""))
    return rows


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'召回':<6} {'排名':<5} {'top1':<7} 问题")
    hit = 0
    for question, needle, note, rank, top1, err in rows:
        if rank:
            hit += 1
        flag = "[OK]  " if rank else "[MISS]"
        rank_text = f"#{rank}" if rank else "-"
        print(f"  {flag} {rank_text:<5} {top1:<7.3f} {question}")
        if not rank:
            print(f"         （top5 里没有含「{needle}」的片段{(' · ' + err) if err else ''}）")
    total = len(rows)
    rate = hit / total * 100 if total else 0
    print(f"  => 命中 {hit}/{total}（{rate:.0f}%）")
    return hit, total


def main() -> int:
    single = run_single()
    multi = run_multi()

    h1, t1 = report("单轮检索", single)
    h2, t2 = report("多轮追问检索", multi)

    print("\n汇总")
    print(f"  单轮     : {h1}/{t1} = {h1 / t1 * 100:.0f}%")
    print(f"  多轮追问 : {h2}/{t2} = {h2 / t2 * 100:.0f}%   ← 当前只用追问原句检索")
    print(f"  合计     : {h1 + h2}/{t1 + t2} = {(h1 + h2) / (t1 + t2) * 100:.0f}%")

    # 多轮全挂是预期内的基线，不作为失败退出；单轮掉到 80% 以下才认为环境有问题
    if t1 and h1 / t1 < 0.8:
        print("\n[警告] 单轮召回低于 80%，先检查知识库是否完整")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
