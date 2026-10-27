"""
问答缓存匹配器评测：词面余弦 / 语义向量 / 交叉编码器重排

目的：回答"除了调低阈值，还有没有更好的办法"。
做法：准备一批人工标注的问题对（同一个问题 / 不是同一个问题），
      分别用四种打分方式算相似度，比较它们的精确率-召回率曲线。
      在同样的精确率下，谁的召回率高，谁的匹配器就更好。

关键指标不是"命中率"本身，而是：
    在"几乎不会给错答案"（精确率 >= 99%）的前提下，能命中多少

因为一次错误的缓存命中 = 用户拿到一个自信但错误的答案，自己无法察觉；
而一次未命中只是多等几秒。

用法：
    cd backend
    .\\venv\\Scripts\\python.exe evaluate_cache_matchers.py
"""
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from app.services.bm25 import BM25Index  # noqa: E402
from app.services.embedding_service import embedding_service  # noqa: E402
from app.utils.chinese_tokenizer import digits, has_negation, tokenize  # noqa: E402

RERANKER_PATH = os.environ.get(
    "RERANKER_PATH", str(Path(__file__).resolve().parents[1] / "models" / "bge-reranker-large")
)

# (标签, 类别, 问题 A（已在缓存里）, 问题 B（新提问）)
# True = 同一个问题，缓存答案可以复用
PAIRS = [
    # ---------- 正例：同一个问题 ----------
    (True, "完全相同", "Milvus 用的是什么索引？", "Milvus 用的是什么索引？"),
    (True, "标点不同", "知识库支持哪些文件格式？", "知识库支持哪些文件格式"),
    (True, "标点不同", "问答缓存怎么失效？", "问答缓存怎么失效"),
    (True, "停用词差异", "这个项目的技术架构是什么", "项目的技术架构是怎样的"),
    (True, "停用词差异", "图谱里的节点是怎么去重的", "图谱节点如何去重"),
    (True, "语序不同", "向量数据库用的是什么", "用的是哪个向量数据库"),
    (True, "语序不同", "分块大小怎么设置", "怎么设置分块大小"),
    (True, "同义改写", "知识库支持上传哪些文件格式", "能上传什么格式的文档"),
    (True, "同义改写", "问答缓存是怎么命中的", "缓存什么情况下会命中"),
    (True, "同义改写", "部署这个系统需要哪些依赖", "运行起来要装什么东西"),
    (True, "口语表达", "这玩意儿咋部署", "部署方式是怎样的"),
    (True, "长短句", "我想知道这份周报里下周计划安排了哪些事情", "周报的下周计划"),
    (True, "长短句", "请问一下知识图谱这个功能是怎么实现的", "知识图谱怎么实现"),
    (True, "同义改写", "Milvus 用的是什么索引", "Milvus 用的哪种索引类型"),
    (True, "同义改写", "文档是怎么切分成块的", "文本是怎样进行分块的"),
    (True, "同义改写", "问答的结果支持流式输出吗", "回答能一个字一个字地出来吗"),
    (True, "同义改写", "怎么删除一篇已经上传的文档", "上传的文档如何移除"),
    (True, "同义改写", "系统里有哪些模型", "系统用到了哪些 AI 模型"),
    (True, "同义改写", "BGE 模型在哪里配置", "BGE 模型的路径在哪个文件里设置"),
    (True, "同义改写", "DeepSeek 的 API Key 在哪里配", "DeepSeek 密钥配置在什么地方"),
    (True, "同义改写", "数据库用的是 MySQL 还是别的", "底层数据库用的是什么"),
    (True, "同义改写", "前端用什么框架做的", "前端用的是什么技术栈"),
    (True, "同义改写", "相似度阈值是多少", "命中阈值设置成多少"),
    (True, "同义改写", "会话历史最多保留多少条", "历史记录能存多少"),
    (True, "同义改写", "系统怎么跑起来", "如何启动这个系统"),
    (True, "同义改写", "有没有现成的接口文档", "API 文档在哪里"),
    (True, "同义改写", "缓存命中之后还会调用大模型吗", "走缓存的时候还需要请求模型吗"),
    (True, "同义改写", "图谱支持增量更新吗", "新增文档后图谱会自动更新吗"),

    # ---------- 负例：完全无关 ----------
    (False, "完全无关", "下周计划做什么", "这个项目的技术架构是什么"),
    (False, "完全无关", "今天天气怎么样", "Milvus 用的是什么索引"),
    (False, "完全无关", "怎么给员工报销", "文档是怎么切分成块的"),

    # ---------- 负例：同主题不同问题（最难区分） ----------
    (False, "同主题不同问题", "Milvus 用的是什么索引", "Milvus 的相似度度量方式是什么"),
    (False, "同主题不同问题", "文档分块大小是多少", "向量维度是多少"),
    (False, "同主题不同问题", "问答缓存存在哪里", "问答缓存什么时候失效"),
    (False, "同主题不同问题", "BGE 模型在哪里配置", "BGE 模型的向量维度是多少"),
    (False, "同主题不同问题", "前端怎么启动", "后端怎么启动"),
    (False, "同主题不同问题", "怎么上传文档", "怎么删除文档"),
    (False, "同主题不同问题", "知识库支持哪些文件格式", "知识库支持多大的文件"),
    (False, "同主题不同问题", "图谱的边是怎么连的", "图谱的节点是怎么去重的"),
    (False, "同主题不同问题", "会话历史保存在哪里", "会话历史怎么清除"),
    (False, "同主题不同问题", "MySQL 端口是多少", "Milvus 端口是多少"),
    (False, "同主题不同问题", "文档支持哪些格式", "文档支持多大的体积"),
    (False, "同主题不同问题", "如何提高检索的召回率", "如何提高检索的准确率"),

    # ---------- 负例：词面极像但语义不同（真正的坑） ----------
    (False, "数字不同", "2024年的家庭预算是多少", "2025年的家庭预算是多少"),
    (False, "数字不同", "分块大小是 500 字吗", "分块大小是 1000 字吗"),
    (False, "肯定/否定", "支持 PDF 上传吗", "不支持 PDF 上传吗"),
    (False, "肯定/否定", "问答缓存可以关闭吗", "问答缓存不能关闭吗"),
    (False, "主体不同", "BGE 模型怎么配置", "DeepSeek 模型怎么配置"),
    (False, "主体不同", "Milvus 的集合名是什么", "MySQL 的表名是什么"),
    (False, "动作不同", "上传文档有哪些限制", "删除文档有哪些限制"),
    (False, "动作不同", "怎么导出问答记录", "怎么导入问答记录"),
    (False, "属性不同", "图谱支持增量更新吗", "图谱支持批量导入吗"),
    (False, "属性不同", "问答缓存的相似度阈值是多少", "文件名匹配的相似度阈值是多少"),
    (False, "主体不同", "前端的技术栈是什么", "后端的技术栈是什么"),
    (False, "属性不同", "文档摘要怎么生成", "文档向量怎么生成"),
    (False, "属性不同", "会话最多保存多少条", "缓存最多保存多少条"),
    (False, "属性不同", "这个系统支持多用户吗", "这个系统支持多语言吗"),
    (False, "属性不同", "支持图片识别吗", "支持表格识别吗"),
    (False, "属性不同", "响应速度怎么样", "准确率怎么样"),
]

GUARD_DIGIT = True   # 数字必须一致
GUARD_NEG = True     # 否定词必须一致


def load_reranker():
    """加载本地 bge-reranker-large（交叉编码器）；没有就返回 None"""
    import torch
    from sentence_transformers import CrossEncoder

    if not os.path.isdir(RERANKER_PATH):
        print(f"  [跳过] 未找到重排模型：{RERANKER_PATH}")
        return None, None
    t0 = time.perf_counter()
    model = CrossEncoder(
        RERANKER_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=128,
    )
    model.predict([("测试问题", "测试问题")], batch_size=1)  # 预热
    return model, (time.perf_counter() - t0) * 1000


def build_scores():
    """返回 matchers{名称: (分数数组, 单次耗时 ms)}, labels, guard_ok"""
    questions, pos = [], {}
    for _, _, a, b in PAIRS:
        for q in (a, b):
            if q not in pos:
                pos[q] = len(questions)
                questions.append(q)

    # ---- 1. 语义向量 ----
    t0 = time.perf_counter()
    vectors = np.asarray(embedding_service.encode(questions), dtype=np.float32)
    emb = np.array([float(vectors[pos[a]] @ vectors[pos[b]]) for _, _, a, b in PAIRS],
                   dtype=np.float32)
    t0 = time.perf_counter()
    embedding_service.encode("预热用的短问题")
    emb_ms = (time.perf_counter() - t0) * 1000

    # ---- 2. 词面加权余弦 ----
    index = BM25Index([tokenize(q) for q in questions])
    t0 = time.perf_counter()
    lex = np.array([index.similarity(pos[a], tokenize(b)) for _, _, a, b in PAIRS],
                   dtype=np.float32)
    lex_ms = (time.perf_counter() - t0) * 1000 / len(PAIRS)

    matchers = {"词面加权余弦": (lex, lex_ms), "语义向量": (emb, emb_ms)}

    # ---- 3. 交叉编码器重排 ----
    reranker, load_ms = load_reranker()
    if reranker is not None:
        raw = reranker.predict([(a, b) for _, _, a, b in PAIRS], batch_size=16)
        # bge-reranker 的 predict 已经返回 0~1 概率，不要再套 sigmoid：
        # 再套一次会把分数压到 0.5~0.73，排序不变但"0.995"这种工作点就丢了
        rer = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), 0.0, 1.0)
        t0 = time.perf_counter()
        for _, _, a, b in PAIRS[:10]:
            reranker.predict([(a, b)], batch_size=1)
        rer_ms = (time.perf_counter() - t0) * 1000 / 10
        matchers["交叉编码器重排"] = (rer, rer_ms)
        print(f"  重排模型加载 {load_ms:.0f} ms")

    labels = np.array([p[0] for p in PAIRS])
    guard_ok = np.ones(len(PAIRS), dtype=bool)
    for i, (_, _, a, b) in enumerate(PAIRS):
        if GUARD_DIGIT and digits(a) != digits(b):
            guard_ok[i] = False
        if GUARD_NEG and has_negation(a) != has_negation(b):
            guard_ok[i] = False

    return matchers, labels, guard_ok, len(questions)


def pr_curve(scores, labels, guard_ok):
    rows = []
    n_pos = int(labels.sum())
    for t in np.arange(1.0, 0.0, -0.005):
        hit = (scores >= t) & guard_ok
        tp = int((hit & labels).sum())
        fp = int((hit & ~labels).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rows.append((float(t), prec, tp / n_pos, tp, fp, n_pos - tp))
    return rows


def best_at(rows, min_precision):
    ok = [r for r in rows if r[1] >= min_precision]
    return max(ok, key=lambda r: (r[2], r[0])) if ok else None


def tiered_metrics(
    lex,
    emb,
    rer,
    labels,
    guard_ok,
    lex_th=0.85,
    sem_th=0.70,
    rer_th=0.995,
):
    """
    线上三层判定的召回率 / 精确率

        命中 = 守卫通过 且 (词面 >= lex_th 或 (语义 >= sem_th 且 重排 >= rer_th))

    rer_th 传 1.01 表示"关掉重排层"，用来还原改动前的基线。
    """
    hit = ((lex >= lex_th) | ((emb >= sem_th) & (rer >= rer_th))) & guard_ok
    tp = int((hit & labels).sum())
    fp = int((hit & ~labels).sum())
    n_pos = int(labels.sum())
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / n_pos if n_pos else 0.0
    return recall, precision, tp, fp


def main() -> int:
    print("=" * 98)
    print("问答缓存匹配器评测")
    print("=" * 98)
    matchers, labels, guard_ok, n_q = build_scores()
    n_pos = int(labels.sum())
    print(f"  问题对 {len(PAIRS)} 条：正例 {n_pos} / 负例 {len(PAIRS) - n_pos}")
    print(f"  涉及不同问句 {n_q} 个；数字守卫=开，否定词守卫=开")

    # ---------------- 一、各自的最优工作点 ----------------
    print()
    print("=" * 98)
    print("一、在「精确率 ≥ 99%」约束下，各匹配器能命中多少")
    print("=" * 98)
    print(f"{'匹配器':<20}{'阈值':>9}{'召回率':>10}{'TP':>6}{'FP':>6}{'漏':>6}{'单次耗时':>12}")
    print("-" * 98)
    curves = {}
    for name, (scores, ms) in matchers.items():
        rows = pr_curve(scores, labels, guard_ok)
        curves[name] = rows
        r = best_at(rows, 0.99)
        if r:
            print(f"{name:<20}{r[0]:>9.3f}{r[2]:>10.1%}{r[3]:>6}{r[4]:>6}{r[5]:>6}{ms:>10.1f}ms")
    print("-" * 98)
    print("  召回率 = 28 条「同一个问题」里，有多少被正确命中")

    # ---------------- 二、放宽精确率看曲线 ----------------
    print()
    print("=" * 98)
    print("二、放宽精确率后能换到多少召回率（说明这不是「调阈值」能解决的）")
    print("=" * 98)
    names = list(curves)
    print(f"{'精确率门槛':<12}" + "".join(f"{n:>16}" for n in names))
    print("-" * 98)
    for p in (0.99, 0.97, 0.95, 0.90, 0.85, 0.80):
        cells = []
        for n in names:
            r = best_at(curves[n], p)
            cells.append(f"{r[2]:>15.1%}" if r else f"{'-':>15}")
        print(f"{p:<12.0%}" + "".join(cells))
    print("-" * 98)
    print("  看最后一行：把精确率降到 80%（等于每 5 次命中就有 1 次答错），")
    print("  召回率仍然上不去多少 —— 说明瓶颈不在阈值，在打分方式本身。")

    # ---------------- 三、误差分析 ----------------
    print()
    print("=" * 98)
    print("三、误差分析：99% 精确率工作点上，漏掉的是哪些")
    print("=" * 98)
    for name, (scores, _) in matchers.items():
        r = best_at(curves[name], 0.99)
        if not r:
            continue
        hit = (scores >= r[0]) & guard_ok
        miss = [i for i in range(len(PAIRS)) if labels[i] and not hit[i]]
        wrong = [i for i in range(len(PAIRS)) if (not labels[i]) and hit[i]]
        print(f"\n  【{name}】阈值 {r[0]:.3f}  漏 {len(miss)} 条 / 错放 {len(wrong)} 条")
        seen = set()
        for i in miss:
            k = PAIRS[i][1]
            if k in seen:
                continue
            seen.add(k)
            print(f"    漏 [{k}] {PAIRS[i][2]}  /  {PAIRS[i][3]}")

    # ---------------- 四、线上三层判定的整体效果 ----------------
    print()
    print("=" * 98)
    print("四、线上三层判定的整体效果（与 qa_cache_service.lookup_ex 一致）")
    print("=" * 98)
    if "交叉编码器重排" in matchers:
        lex = matchers["词面加权余弦"][0]
        emb = matchers["语义向量"][0]
        rer = matchers["交叉编码器重排"][0]
        print(f"{'方案':<42}{'召回率':>10}{'精确率':>10}{'TP':>6}{'FP':>6}")
        print("-" * 98)
        for label, rer_th in (
            ("① 仅词面 ≥ 0.85（改动前基线）", 1.01),  # 1.01 恒不成立 = 关掉重排层
            ("② 词面 或 (语义 ≥0.70 且 重排 ≥0.995)", 0.995),
            ("③ 参考：重排阈值放宽到 0.98", 0.98),
            ("④ 参考：重排阈值放宽到 0.95", 0.95),
        ):
            rec, prec, tp, fp = tiered_metrics(lex, emb, rer, labels, guard_ok, rer_th=rer_th)
            print(f"{label:<42}{rec:>10.1%}{prec:>10.1%}{tp:>6}{fp:>6}")
        print("-" * 98)
        print(f"  重排单次推理 {matchers['交叉编码器重排'][1]:.1f} ms；"
              f"数字/否定词不同的 {int((~guard_ok).sum())} 条被守卫直接排除")

    # ---------------- 五、正例分布 ----------------
    print()
    print("=" * 98)
    print("五、28 条正例上的分数分布（越低说明越容易漏）")
    print("=" * 98)
    for name, (scores, _) in matchers.items():
        v = scores[labels]
        print(
            f"  {name:<16} 均值 {v.mean():.3f}  中位数 {np.median(v):.3f}  "
            f"最小 {v.min():.3f}  10分位 {np.percentile(v, 10):.3f}  "
            f"最大 {v.max():.3f}"
        )
    for name, (scores, _) in matchers.items():
        v = scores[~labels]
        print(f"  {name:<16} （负例）均值 {v.mean():.3f}  最大 {v.max():.3f}")
    print()
    print("  两组分布重叠越多，任何单一阈值都分不开 —— 这就是问题的本质。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
