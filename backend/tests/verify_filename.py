"""
文件名匹配验证脚本

覆盖三件事：
    A. 名称匹配算法本身（纯函数，不需要服务）
    B. 知识图谱的文件名关联边（纯函数 + 接口）
    C. 端到端：文件名召回 / 改名 / 问答引用里的文件名

用法（需要后端已启动）：
    cd backend
    .\\venv\\Scripts\\python.exe verify_filename.py
"""

import sys
from pathlib import Path

# 本文件在 backend/tests/ 下，把 backend/ 加进 sys.path 才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import json
import sys
import time
import uuid

import httpx

BACKEND = "http://127.0.0.1:8000/api/v1"

passed = []
failed = []


def check(name, ok, detail=""):
    if ok:
        passed.append(name)
        print(f"  [OK]   {name} {detail}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name} {detail}")


# ======================= A. 名称匹配算法 =======================
def test_matcher():
    from app.utils.name_matcher import (
        NameKey,
        build_embed_text,
        is_placeholder_name,
        match_score,
        similarity,
        similarity_between,
    )

    print("\n[A1] 提问-文件名相似度")

    cases = [
        # (提问, 文件名, 期望下限, 期望上限, 说明)
        ("作业2-王小明", "作业2-王小明.pdf", 1.0, 1.0, "与文件名完全一致"),
        ("王小明的作业2结论是什么", "作业2-王小明.pdf", 1.0, 1.0, "所有片段都出现"),
        ("周报", "周报-知识库项目进展.md", 0.90, 1.0, "提问是文件名的片段"),
        ("周报里写了什么", "周报-知识库项目进展.md", 0.60, 1.0, "提问包含片段"),
        ("知识库项目进展的周报", "周报-知识库项目进展.md", 1.0, 1.0, "片段全部命中"),
        ("家庭年度预算怎么规划", "家庭年度预算与理财计划.md", 0.60, 1.0, "长公共子串"),
        (
            "手写数字识别的结论",
            "基于人工神经网络的手写数字识别技术报告.pdf",
            0.60,
            1.0,
            "从长标题里认出关键短语",
        ),
        ("项目", "周报-知识库项目进展.md", 0.0, 0.30, "泛词不应命中"),
        ("这个项目的部署方式是什么", "周报-知识库项目进展.md", 0.0, 0.44, "普通提问不应命中"),
        ("今天天气怎么样", "作业2-王小明.pdf", 0.0, 0.0, "完全无关"),
    ]
    for query, filename, low, high, desc in cases:
        score = similarity(NameKey(query), NameKey(filename))
        check(f"{desc}", low <= score <= high, f"{query!r} vs {filename!r} -> {score}")

    print("\n[A2] 文件名 + 标题联合匹配")
    score, field = match_score(
        NameKey("王小明"), "349c0426c6244e3397ea1cf236f6cec0.pdf", "王小明"
    )
    check("标题命中时取标题", field == "title" and score >= 0.9, f"{score} / {field}")

    score, field = match_score(NameKey("作业2-王小明"), "作业2-王小明.pdf", "手写数字识别")
    check("文件名优先", field == "filename" and score == 1.0, f"{score} / {field}")

    print("\n[A3] 占位名识别")
    check(
        "UUID 占位名",
        is_placeholder_name("a84b6fef40264cdfa2db8cf839d6dece.md")
        and not is_placeholder_name("作业2-王小明.pdf"),
    )

    print("\n[A4] 嵌入文本带文件名")
    text = build_embed_text("作业2-王小明.pdf", "手写数字识别技术报告", "正文内容")
    check(
        "文件名与标题拼进嵌入文本",
        "作业2-王小明" in text and "手写数字识别技术报告" in text and text.endswith("正文内容"),
        repr(text[:50]),
    )
    text = build_embed_text("a84b6fef40264cdfa2db8cf839d6dece.md", None, "正文")
    check("占位名不进嵌入文本", "a84b6fef" not in text, repr(text))

    print("\n[A5] 两个名字之间的相似度（图谱用）")
    check(
        "同系列文档",
        similarity_between("作业2-王小明.pdf", "作业2-王小明-修订版.pdf") >= 0.6,
    )
    check(
        "不同文档",
        similarity_between("作业2-王小明.pdf", "家庭年度预算与理财计划.md") < 0.3,
    )


# ======================= B. 图谱文件名关联边 =======================
def test_graph_edges():
    from app.agents.relation_agent import RelationAgent

    print("\n[B1] 文件名关联边（构造数据）")
    rows = [
        {"id": 901, "name": "作业2-王小明", "title": "作业2-王小明", "filename": "作业2-王小明.pdf"},
        {
            "id": 902,
            "name": "作业2-王小明-修改版",
            "title": "作业2-王小明-修改版",
            "filename": "作业2-王小明-修改版.pdf",
        },
        {"id": 903, "name": "咖啡冲煮笔记", "title": "咖啡冲煮笔记", "filename": "咖啡冲煮笔记.md"},
    ]
    edges = []
    added = RelationAgent._attach_name_edges(rows, edges, 0.6)
    check("生成 1 条文件名关联边", added == 1, f"added={added}")
    check(
        "边为虚线且标记 relation=filename",
        len(edges) == 1
        and edges[0]["relation"] == "filename"
        and edges[0]["lineStyle"]["type"] == "dashed",
        json.dumps(edges[0] if edges else {}, ensure_ascii=False),
    )
    check("无关文档不连边", all("903" not in (e["source"], e["target"]) for e in edges))

    print("\n[B2] 阈值 0 = 关闭")
    edges = []
    added = RelationAgent._attach_name_edges(rows, edges, 0)
    check("关闭后不生成", added == 0 and not edges, f"added={added}")

    print("\n[B3] 已有语义边时只补 name_score 并改虚线")
    edges = [
        {
            "source": "901",
            "target": "902",
            "value": 0.5,
            "lineStyle": {"width": 1.0},
        }
    ]
    added = RelationAgent._attach_name_edges(rows, edges, 0.6)
    check(
        "不重复连边",
        added == 0 and edges[0].get("relation") == "both" and edges[0].get("name_score", 0) > 0.6,
        json.dumps(edges[0], ensure_ascii=False),
    )
    check(
        "语义边也改成虚线",
        edges[0].get("lineStyle", {}).get("type") == "dashed",
        json.dumps(edges[0].get("lineStyle"), ensure_ascii=False),
    )


# ======================= C. 端到端 =======================
def test_api():
    from app.db.database import session_scope
    from app.models.document import Document
    from app.utils.name_matcher import is_placeholder_name

    token = uuid.uuid4().hex[:8]
    real_name = f"文件名匹配测试-{token}.md"
    # 改名时换一个全新的随机串：否则新旧名字共享同一个 token，
    # 名称匹配照样会命中，测不出"旧名字失效"
    renamed = f"改名后测试-{uuid.uuid4().hex[:8]}.md"
    renamed_stem = renamed.rsplit(".", 1)[0]
    content = (
        "# 咖啡冲煮笔记\n\n"
        "手冲咖啡建议水温 92 度，粉水比 1:15，闷蒸 30 秒。\n"
        "研磨度接近白砂糖，注水三段完成，总时长约两分半。\n"
        f"\n<!-- {token} -->\n"
    )

    doc_id = None
    with httpx.Client(timeout=180.0) as c:
        print("\n[C1] 现有文档的文件名已就位")
        r = c.get(f"{BACKEND}/documents", params={"size": 100})
        items = r.json().get("items", [])
        placeholders = [i["id"] for i in items if is_placeholder_name(i.get("filename"))]
        check(
            "没有 UUID 占位文件名",
            not placeholders,
            f"共 {len(items)} 篇，占位名 {placeholders}",
        )
        missing = [i["filename"] for i in items if not i.get("filename")]
        check("每篇文档都有文件名", not missing, str(missing[:3]))

        print("\n[C2] Milvus 里的 filename 与 MySQL 一致")
        from app.services.milvus_store import milvus_store

        with session_scope() as db:
            mysql_map = {d.id: d.filename for d in db.query(Document).all()}
        rows = milvus_store.collection.query(
            expr="pk >= 0", output_fields=["document_id", "filename"], limit=500
        )
        milvus_map = {}
        for row in rows:
            milvus_map.setdefault(int(row["document_id"]), set()).add(row.get("filename"))
        mismatched = [
            (doc_id, sorted(names), mysql_map.get(doc_id))
            for doc_id, names in milvus_map.items()
            if mysql_map.get(doc_id) not in names
        ]
        check(
            "向量库 filename 与数据库一致",
            not mismatched,
            f"检查 {len(milvus_map)} 篇，异常 {mismatched[:3]}",
        )

        # ---------- 上传一篇行为"只靠名字才找得到"的文档 ----------
        print("\n[C3] 上传测试文档（关掉 AI 摘要，加快速度）")
        files = {"file": (real_name, io.BytesIO(content.encode("utf-8")), "text/markdown")}
        r = c.post(f"{BACKEND}/documents/upload", files=files, params={"generate_summary": False})
        check("上传成功", r.status_code == 200, r.json().get("message", ""))
        if r.status_code != 200:
            return

        deadline = time.time() + 180
        status = None
        while time.time() < deadline:
            items = c.get(f"{BACKEND}/documents", params={"size": 100}).json().get("items", [])
            target = next((i for i in items if i["filename"] == real_name), None)
            if target:
                doc_id = target["id"]
                status = target["status"]
                if status in ("ready", "failed"):
                    break
            time.sleep(2)
        check("索引完成", status == "ready", f"doc_id={doc_id} status={status}")
        if status != "ready":
            return

        print("\n[C4] 按文件名检索能命中")
        r = c.post(f"{BACKEND}/search", json={"query": f"文件名匹配测试-{token}", "top_k": 5})
        data = r.json()
        names = [m["document_id"] for m in data.get("name_matches", [])]
        check("name_matches 包含该文档", doc_id in names, f"name_matches={names}")
        hit_docs = [d["document_id"] for d in data.get("documents", [])]
        check("检索结果包含该文档", doc_id in hit_docs, f"documents={hit_docs}")
        top = data["documents"][0] if data.get("documents") else {}
        check(
            "该文档排在第一",
            top.get("document_id") == doc_id,
            f"top={top.get('document_id')} score={top.get('best_score')} type={top.get('match_type')}",
        )
        chunk = next((x for x in data.get("chunks", []) if x["document_id"] == doc_id), {})
        check(
            "命中片段的文件名正确",
            chunk.get("filename") == real_name,
            f"{chunk.get('filename')} match_type={chunk.get('match_type')}",
        )
        score_on = chunk.get("score")
        name_score_on = chunk.get("name_score")
        vector_score_on = chunk.get("vector_score")

        print("\n[C5] 关掉文件名匹配后分数回落（说明名称通道确实在加分）")
        r = c.post(
            f"{BACKEND}/search",
            json={"query": f"文件名匹配测试-{token}", "top_k": 5, "use_name_match": False},
        )
        data = r.json()
        check("关闭后无 name_matches", not data.get("name_matches"), str(data.get("name_matches")))
        chunk_off = next(
            (x for x in data.get("chunks", []) if x["document_id"] == doc_id), {}
        )
        score_off = chunk_off.get("score")
        check(
            "关闭后分数变低",
            score_off is not None and score_on is not None and score_off < score_on,
            f"{score_on} -> {score_off}",
        )
        # 分差应当等于 名称相似度 × filename_boost(0.2)
        expected = round(0.2 * (name_score_on or 0), 4)
        actual = round((score_on or 0) - (score_off or 0), 4)
        check(
            "分差等于名称加分",
            abs(actual - expected) <= 0.002,
            f"实际 {actual} / 期望 {expected}（名称相似度 {name_score_on}，向量分 {vector_score_on}）",
        )

        print("\n[C6] 改名后能按新名字搜到")
        r = c.patch(f"{BACKEND}/documents/{doc_id}", json={"filename": renamed})
        check(
            "改名接口",
            r.status_code == 200 and r.json().get("filename") == renamed,
            r.json().get("filename", r.text[:80]),
        )

        r = c.post(f"{BACKEND}/search", json={"query": renamed_stem, "top_k": 5})
        data = r.json()
        names = [m["document_id"] for m in data.get("name_matches", [])]
        hit_docs = [d["document_id"] for d in data.get("documents", [])]
        check("新名字命中", doc_id in names and doc_id in hit_docs, f"names={names} docs={hit_docs}")
        chunk = next((x for x in data.get("chunks", []) if x["document_id"] == doc_id), {})
        check(
            "Milvus 中的 filename 同步更新",
            chunk.get("filename") == renamed,
            f"{chunk.get('filename')}",
        )

        r = c.post(f"{BACKEND}/search", json={"query": f"文件名匹配测试-{token}", "top_k": 5})
        names = [m["document_id"] for m in r.json().get("name_matches", [])]
        check("旧名字不再命中", doc_id not in names, f"names={names}")

        print("\n[C7] 问答引用里带文件名")
        r = c.post(
            f"{BACKEND}/qa",
            json={
                "question": f"{renamed_stem} 里咖啡水温是多少？",
                "top_k": 3,
                "save_history": False,
                "use_cache": False,
            },
        )
        data = r.json()
        cites = data.get("citations", [])
        check("有引用", bool(cites), f"引用 {len(cites)} 条")
        if cites:
            check(
                "引用带 filename 字段",
                all("filename" in x for x in cites) and bool(cites[0].get("filename")),
                f"{cites[0].get('filename')} title={cites[0].get('title')}",
            )
            check(
                "引用命中该文档",
                cites[0].get("document_id") == doc_id,
                f"doc={cites[0].get('document_id')} type={cites[0].get('match_type')}",
            )
        print("    回答:", (data.get("answer") or "")[:120].replace("\n", " "))

        print("\n[C8] 图谱节点带文件名")
        g = c.get(f"{BACKEND}/graph", params={"min_score": 0.45}).json()
        check(
            "每个节点都有 filename",
            all(n.get("filename") for n in g.get("nodes", [])),
            f"节点 {g.get('count')} 个",
        )
        g2 = c.get(f"{BACKEND}/graph", params={"min_score": 0.45, "name_min_score": 0}).json()
        check(
            "关闭文件名关联后 name_edge_count=0",
            g2.get("name_edge_count") == 0,
            f"name_edges={g2.get('name_edge_count')} edges={len(g2.get('edges', []))}",
        )

        print("\n[C9] 清理")
        r = c.delete(f"{BACKEND}/documents/{doc_id}")
        check("删除测试文档", r.status_code == 200, r.json().get("message", ""))


def main() -> int:
    try:
        test_matcher()
        test_graph_edges()
        test_api()
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
