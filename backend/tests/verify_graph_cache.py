"""
知识图谱向量缓存验证

要证明四件事：
    A. 第一次打开图谱：全部编码
    B. 再打开（含拖动阈值滑块）：全部命中缓存，耗时明显下降
    C. 新增一篇文档：只编码这一篇，其余复用
    D. 删除该文档：缓存行随之删除（外键级联），图谱结果回到原样

用法（需要后端已启动）：
    cd backend
    .\\venv\\Scripts\\python.exe verify_graph_cache.py
"""
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


def get_graph(c: httpx.Client, **params) -> dict:
    r = c.get(f"{BACKEND}/graph", params={"min_score": 0.45, **params})
    r.raise_for_status()
    return r.json()


def main() -> int:
    token = uuid.uuid4().hex[:8]
    doc_name = f"图谱缓存测试-{token}.md"
    content = (
        f"# 图谱缓存测试 {token}\n\n"
        "本文档用于验证知识图谱的文档向量缓存是否只重算新增文档。\n"
        "内容刻意与知识库其它文档不相关：紫砂壶的开壶与养壶方法。\n"
        f"\n<!-- {token} -->\n"
    ) * 3

    with httpx.Client(timeout=180.0) as c:
        print("\n[A] 第一次计算（缓存为空或部分为空）")
        first = get_graph(c, force_refresh=True)
        stats = first.get("vector_stats") or {}
        check("接口返回 vector_stats", bool(stats), str(stats))
        check(
            "强制刷新时全部重新编码",
            stats.get("encoded") == stats.get("documents") and stats.get("reused") == 0,
            f"encoded={stats.get('encoded')} reused={stats.get('reused')}",
        )
        base_nodes = first.get("count")
        base_edges = len(first.get("edges") or [])
        print(f"    节点 {base_nodes} / 边 {base_edges}")

        print("\n[B] 再次计算（应全部命中缓存）")
        t0 = time.perf_counter()
        second = get_graph(c)
        elapsed_cached = (time.perf_counter() - t0) * 1000
        stats2 = second.get("vector_stats") or {}
        check("全部复用", stats2.get("encoded") == 0 and stats2.get("reused") > 0, str(stats2))
        check(
            "两次结果一致",
            second.get("count") == base_nodes and len(second.get("edges") or []) == base_edges,
            f"{second.get('count')} / {len(second.get('edges') or [])}",
        )

        print("\n[B2] 拖动阈值滑块（min_score 变化，向量不该重算）")
        t0 = time.perf_counter()
        slid = get_graph(c, min_score=0.6)
        elapsed_slider = (time.perf_counter() - t0) * 1000
        stats3 = slid.get("vector_stats") or {}
        check("换阈值仍然全部复用", stats3.get("encoded") == 0, str(stats3))
        check(
            "阈值生效（边变少）",
            len(slid.get("edges") or []) <= base_edges,
            f"{base_edges} -> {len(slid.get('edges') or [])} 条边",
        )
        print(f"    本次请求 {elapsed_slider:.0f}ms（含读库+矩阵+连边）")
        second_stats = second.get("vector_stats") or {}
        print(
            f"    编码耗时对比：冷 {first.get('vector_stats', {}).get('encode_ms')}ms"
            f" -> 热 {second_stats.get('encode_ms')}ms"
        )

        print("\n[C] 新增一篇文档（只应编码它自己）")
        files = {"file": (doc_name, io.BytesIO(content.encode("utf-8")), "text/markdown")}
        r = c.post(f"{BACKEND}/documents/upload", files=files, params={"generate_summary": False})
        check("上传成功", r.status_code == 200, r.json().get("message", ""))

        doc_id = None
        deadline = time.time() + 120
        while time.time() < deadline:
            items = c.get(f"{BACKEND}/documents", params={"size": 100}).json().get("items", [])
            target = next((i for i in items if i["filename"] == doc_name), None)
            if target and target["status"] in ("ready", "failed"):
                doc_id = target["id"]
                break
            time.sleep(1.5)
        check("索引完成", doc_id is not None, f"doc_id={doc_id}")

        after_add = get_graph(c)
        stats4 = after_add.get("vector_stats") or {}
        check(
            "只编码新文档",
            stats4.get("encoded") == 1,
            f"encoded={stats4.get('encoded')} reused={stats4.get('reused')} "
            f"documents={stats4.get('documents')}",
        )
        check(
            "节点数 +1",
            after_add.get("count") == base_nodes + 1,
            f"{base_nodes} -> {after_add.get('count')}",
        )

        print("\n[C2] 再算一次（新文档也进了缓存）")
        again = get_graph(c)
        stats5 = again.get("vector_stats") or {}
        check("编码 0 篇", stats5.get("encoded") == 0, str(stats5))

        print("\n[D] 删除文档后缓存行应随之消失")
        before = c.get(f"{BACKEND}/graph/vector-cache").json()
        c.delete(f"{BACKEND}/documents/{doc_id}")
        time.sleep(1.0)
        after = c.get(f"{BACKEND}/graph/vector-cache").json()
        check(
            "缓存行减少 1",
            after.get("documents") == before.get("documents") - 1,
            f"{before.get('documents')} -> {after.get('documents')}",
        )
        restored = get_graph(c)
        check(
            "图谱回到原样",
            restored.get("count") == base_nodes
            and len(restored.get("edges") or []) == base_edges,
            f"{restored.get('count')} / {len(restored.get('edges') or [])}",
        )
        check(
            "删除后不需要重算其它文档",
            (restored.get("vector_stats") or {}).get("encoded") == 0,
            str(restored.get("vector_stats")),
        )

        print("\n[E] 缓存概况")
        info = c.get(f"{BACKEND}/graph/vector-cache").json()
        check("有缓存统计", info.get("documents", 0) >= 1, str(info))
        print(
            f"    {info.get('documents')} 篇 / {info.get('bytes', 0) / 1024:.0f} KB"
            f" / {info.get('model')}"
        )

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
