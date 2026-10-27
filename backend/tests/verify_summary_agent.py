"""
摘要 Agent 的因果验证（不启动后端，直接调 Agent）

和"跑通就行"的冒烟测试不同，这里要证明的是**改动的因果关系**：

    旧实现固定只读正文前 2500 字，凡是结论埋在 2500 字之后的文档，它必然漏掉。
    本测试造一份约 6500 字的文档，把唯一的关键结论放在最后一段（约 6000 字处），
    然后检查两件事：
        A. 机制：轨迹里确实读了尾部的那几块
        B. 效果：摘要里出现了结论里的关键信息

    两个都成立，才算"把截断换成了有计划地读全文"。

另外验证健壮性：
    C. 步数上限压到 2 时，仍然能产出摘要（reduce 兜底，不会卡在 processing）
    D. 同一范围不会被重复读（防死循环）
    E. 短文档走单次模式，不浪费 token

用法：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\verify_summary_agent.py
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.summary_agent import summary_agent  # noqa: E402
from app.db.database import init_db, session_scope  # noqa: E402
from app.models.document import Chunk, Document  # noqa: E402

MARKER = "ORION-7749"
TRUNCATE_AT = 2500  # 旧实现的截断点

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


# ---------------- 造文档 ----------------
_FILLER_1 = (
    "本报告围绕生产线的设备养护流程展开，先交代背景与调研范围。"
    "调研覆盖三个车间、十七条产线，访谈对象包括班组长的十一位、设备工程师九位。"
    "记录显示，当前养护以固定周期为主，缺少基于状态的调整机制，"
    "导致部分设备过度养护、部分设备养护不足。"
)
_FILLER_2 = (
    "第二阶段对采集到的运行数据做了统计描述。"
    "温度、振动、电流三路信号的缺失率分别为百分之二、百分之五、百分之十二。"
    "振动信号的分布偏态明显，需先做对数变换再进入后续分析。"
    "与历史工单对齐后，共识别出四类典型故障模式，其中轴承磨损占比最高。"
)
_CONCLUSION = (
    "最后给出本阶段的结论与建议。"
    f"综合三路信号与工单记录，最终结论：测试代号 {MARKER}，模型在验证集上的误差率 3.2%。"
    "建议把固定周期养护改为按状态触发的混合策略，优先在振动超标的分组试点。"
    "同时补充缺失率较高的电流通道采集，否则该通道无法参与建模。"
    "以上为本次调研的全部结论。"
)


def build_body() -> str:
    """拼一篇结论在末尾的文档（结论落在 2500 字之后）"""
    part1 = _FILLER_1 * 18          # ≈ 2400 字
    part2 = _FILLER_2 * 18          # ≈ 2400 字
    part3 = _CONCLUSION * 6         # ≈ 1100 字
    return part1 + part2 + part3


def make_chunks(body: str, size: int = 500, overlap: int = 80):
    chunks = []
    start = 0
    while start < len(body):
        chunks.append(body[start : start + size])
        start += size - overlap
    return chunks


def insert_doc(body: str, name: str) -> int:
    pieces = make_chunks(body)
    token = uuid.uuid4().hex
    with session_scope() as db:
        doc = Document(
            filename=name,
            stored_name=f"{token}.md",
            file_path=str(Path("data/documents") / f"{token}.md"),
            file_type="md",
            file_size=len(body.encode("utf-8")),
            file_hash=token + token,
            char_count=len(body),
            chunk_count=len(pieces),
            status="ready",
        )
        db.add(doc)
        db.flush()
        for i, piece in enumerate(pieces):
            db.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=piece,
                    char_count=len(piece),
                )
            )
        doc_id = doc.id
    return doc_id


def read_trace(doc_id: int):
    import json

    with session_scope() as db:
        doc = db.get(Document, doc_id)
        raw = doc.summary_trace if doc else None
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def cleanup(doc_id: int) -> None:
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc is not None:
            db.delete(doc)


def covers_tail(trace, chunk_count: int) -> bool:
    """轨迹里是否有一次读取覆盖到最后一块"""
    for item in trace:
        if item.get("action") != "read_chunks":
            continue
        args = item.get("args") or {}
        try:
            end = int(args.get("end", args.get("start", -1)))
        except (TypeError, ValueError):
            continue
        if end >= chunk_count - 1:
            return True
    return False


def main() -> int:
    init_db()
    body = build_body()
    pos = body.find(MARKER)
    chunk_count = len(make_chunks(body))

    print(f"\n[0] 样本构造（{len(body)} 字 / {chunk_count} 块）")
    check("关键结论位于 2500 字之后（旧实现必漏）", pos > TRUNCATE_AT, f"结论在 {pos} 字处")
    check("文档足够长，会启用 ReAct", chunk_count > 6 and len(body) > 3500)

    doc_id = None
    try:
        doc_id = insert_doc(body, "摘要Agent验证-结论在末尾.md")

        # ---------- A/B：机制 + 效果 ----------
        print("\n[1] 长文档：ReAct 是否真的读了尾部")
        result = summary_agent.run(
            document_id=doc_id,
            filename="摘要Agent验证-结论在末尾.md",
            persist=True,
        )
        trace = read_trace(doc_id)
        actions = [t.get("action") for t in trace]

        check("走了 ReAct 模式", result["mode"] == "react", f"mode={result['mode']}")
        check("轨迹非空", len(trace) > 0, f"{len(trace)} 步")
        check("读取动作覆盖到文档尾部", covers_tail(trace, chunk_count))
        check("调用了 finish 提交", "finish" in actions, str(actions))
        check("步数在上限内", len(trace) <= summary_agent.max_steps + 1, f"{len(trace)} 步")

        summary = result["summary"]
        check("摘要非空", len(summary) > 20, f"{len(summary)} 字")
        hit = MARKER in summary or "3.2" in summary
        check("摘要里出现了结论的关键信息", hit, summary[:80])

        total_chars = max((t.get("chars", 0) for t in trace), default=0)
        check(
            "累计读入未超过预算",
            total_chars <= summary_agent.max_total_chars + 1000,
            f"{total_chars} 字",
        )

        # ---------- D：不重复读 ----------
        print("\n[2] 防止重复读取")
        reads = [t for t in trace if t.get("action") == "read_chunks"]
        keys = [
            (int(t["args"].get("start", -1)), int(t["args"].get("end", -1)))
            for t in reads
            if isinstance(t.get("args"), dict)
        ]
        check("没有重复读取同一范围", len(keys) == len(set(keys)), f"读取 {len(reads)} 次")

        # ---------- C：兜底 ----------
        print("\n[3] 步数压到 2：必须仍能产出摘要")
        saved_steps = summary_agent.max_steps
        summary_agent.max_steps = 2
        try:
            doc2 = insert_doc(body, "摘要Agent验证-兜底.md")
            try:
                r2 = summary_agent.run(document_id=doc2, filename="兜底.md", persist=True)
                t2 = read_trace(doc2)
                check("兜底路径仍产出摘要", len(r2["summary"]) > 20, f"{len(r2['summary'])} 字")
                check(
                    "记录了兜底动作",
                    any(t.get("action") == "reduce_fallback" for t in t2),
                    str([t.get("action") for t in t2]),
                )
            finally:
                cleanup(doc2)
        finally:
            summary_agent.max_steps = saved_steps

        # ---------- E：短文档 ----------
        print("\n[4] 短文档走单次模式")
        short = insert_doc("这是一份很短的家庭预算说明，记录月度收支与储蓄目标。" * 4, "摘要Agent验证-短.md")
        try:
            r3 = summary_agent.run(document_id=short, filename="短.md", persist=True)
            check("短文档走 single 模式", r3["mode"] == "single", f"mode={r3['mode']}")
            check("短文档不产生轨迹", len(r3["trace"]) == 0)
            check("短文档仍有摘要", len(r3["summary"]) > 10)
        finally:
            cleanup(short)

    finally:
        if doc_id:
            cleanup(doc_id)

    print(f"\n通过 {len(passed)} / {len(passed) + len(failed)}")
    if failed:
        print("失败：" + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
