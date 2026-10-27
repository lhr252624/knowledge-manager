"""
用大模型做「这两个问题是不是同一个」的二次判定 —— 实测

背景：单靠一个相似度阈值，召回率提不上去（见 evaluate_cache_matchers.py）。
      那就把"找候选"和"做判定"拆开：候选可以放宽，判定交给大模型。

本脚本在同一批人工标注的问题对上，测三件事：
    1. 大模型判定的准确率 / 精确率 / 召回率
    2. 单次判定耗时（对比未命中时生成答案的 ~7400 ms）
    3. 判错的样本长什么样

用法：
    cd backend
    .\\venv\\Scripts\\python.exe evaluate_llm_judge.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.llm_service import llm_service  # noqa: E402

from evaluate_cache_matchers import PAIRS  # noqa: E402

# deepseek-flash 是推理模型：先输出 reasoning_content 再输出正文。
# max_tokens 太小时思考过程会把预算吃光，content 会是空字符串 ——
# 底层表现为"所有问题都判否"，非常隐蔽。512 足够容纳"是/否"的思考。
MAX_TOKENS = 512

SYSTEM = (
    "你负责判断两个中文问题是不是在问同一件事。\n"
    "判断标准：\n"
    "1. 只是措辞、语序、标点、口语化程度、详略不同 -> 同一件事\n"
    "2. 主体、对象、动作、属性、数字、肯定/否定 有任何一处不同 -> 不是同一件事\n"
    "3. 一个是另一个的上位/下位问题（问的范围不一样）-> 不是同一件事\n"
    "只输出一个字：是 或 否。"
)
PROMPT = "问题A：{a}\n问题B：{b}"


def judge(a: str, b: str) -> tuple[bool, float, str]:
    started = time.perf_counter()
    for _ in range(2):
        reply = llm_service.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(a=a, b=b)},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        ).strip()
        if reply:
            return reply.startswith("是"), (time.perf_counter() - started) * 1000, reply
    # 两次都是空回复：保守判否
    return False, (time.perf_counter() - started) * 1000, ""


def main() -> int:
    print("=" * 96)
    print("大模型二次判定实测")
    print("=" * 96)
    n_pos = sum(1 for p in PAIRS if p[0])
    print(f"  样本 {len(PAIRS)} 对（正例 {n_pos} / 负例 {len(PAIRS) - n_pos}）")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda p: judge(p[2], p[3]), PAIRS))
    wall = time.perf_counter() - started

    preds = [r[0] for r in results]
    lat = sorted(r[1] for r in results)
    empty = sum(1 for r in results if not r[2])

    tp = sum(1 for p, q in zip(PAIRS, preds) if p[0] and q)
    fp = sum(1 for p, q in zip(PAIRS, preds) if not p[0] and q)
    fn = sum(1 for p, q in zip(PAIRS, preds) if p[0] and not q)
    tn = sum(1 for p, q in zip(PAIRS, preds) if not p[0] and not q)

    print()
    print("=" * 96)
    print("一、判定质量")
    print("=" * 96)
    print(
        f"  准确率 {(tp + tn) / len(PAIRS):.1%}   "
        f"精确率 {(tp / (tp + fp) if tp + fp else 1.0):.1%}   "
        f"召回率 {tp / n_pos:.1%}"
    )
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}   空回复 {empty} 次")
    print(
        f"  墙钟 {wall:.1f}s（6 并发）  单次 平均 {sum(lat)/len(lat):.0f} ms / "
        f"P50 {lat[len(lat)//2]:.0f} ms / P90 {lat[int(len(lat)*0.9)]:.0f} ms"
    )

    print()
    print("=" * 96)
    print("二、判错的样本")
    print("=" * 96)
    bad = [(p, q) for p, q in zip(PAIRS, preds) if p[0] != q]
    if not bad:
        print("  （无）")
    for p, q in bad:
        print(f"  [{p[1]}] 标签={'同一件事' if p[0] else '不同'}  判定={'是' if q else '否'}")
        print(f"      A: {p[2]}")
        print(f"      B: {p[3]}")

    print()
    print("=" * 96)
    print("三、和前两种匹配器对比（同一批样本）")
    print("=" * 96)
    print(f"{'方案':<26}{'精确率':>10}{'召回率':>10}{'单次耗时':>14}")
    print("-" * 96)
    print(f"{'词面余弦 ≥0.85（现状）':<26}{'100.0%':>10}{'28.6%':>10}{'~1 ms':>14}")
    print(f"{'交叉编码器重排':<26}{'100.0%':>10}{'39.3%':>10}{'~41 ms':>14}")
    print(
        f"{'大模型判定':<24}"
        f"{(tp / (tp + fp) if tp + fp else 1.0):>10.1%}{tp / n_pos:>10.1%}"
        f"{sum(lat)/len(lat):>12.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
