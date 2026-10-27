"""
HTTP 接口测试

覆盖前端真实调用路径：
    stats -> upload(multipart) -> 轮询状态 -> search -> qa -> qa/stream(SSE)
    -> graph -> delete
并验证 Vite 代理（http://127.0.0.1:3000/api/v1）也能通。

用法：
    cd backend
    .\\venv\\Scripts\\python.exe test_api.py
"""
import io
import json
import sys
import time
import uuid

import httpx

BACKEND = "http://127.0.0.1:8000/api/v1"
PROXY = "http://127.0.0.1:3000/api/v1"

# 固定且独特的文件名，便于在知识库中精确定位本次测试文档
TEST_FILENAME = "自动化测试-周报知识库项目进展.md"

DOC = """# 周报：知识库项目进展

## 本周完成
1. 完成文档解析模块，支持 PDF、Word、Markdown、TXT 四种格式。
2. 接入 BGE-M3 嵌入模型，向量维度 1024，已启用 GPU 加速。
3. Milvus 集合建立完成，使用 HNSW 索引与内积度量。
4. 前端五个页面开发完毕：概览、问答、检索、文档、图谱。

## 下周计划
- 补充批量导入能力，支持整个目录递归索引。
- 增加文档标签体系，便于按主题筛选。
- 优化问答提示词，减少无关引用。

## 风险
GPU 显存 8GB，批量索引大文档时需要控制批大小，避免显存溢出。
"""

passed = []
failed = []


def check(name, ok, detail=""):
    if ok:
        passed.append(name)
        print(f"  [OK]   {name} {detail}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name} {detail}")


def main() -> int:
    doc_id = None

    with httpx.Client(timeout=180.0) as c:
        # ---------- 1. 健康与统计 ----------
        print("\n[1] 基础接口")
        r = c.get(f"{BACKEND}/health/detail")
        check(
            "health/detail",
            r.status_code == 200,
            json.dumps(r.json().get("checks"), ensure_ascii=False),
        )

        r = c.get(f"{BACKEND}/stats")
        check("stats", r.status_code == 200, f"documents={r.json().get('total_documents')}")

        r = c.get(f"{BACKEND}/agents")
        check("agents", r.status_code == 200, f"count={len(r.json().get('agents', []))}")

        # ---------- 2. 上传 ----------
        print("\n[2] 文档上传（multipart）")
        # 每次内容带唯一标记，避免被知识库的哈希去重命中
        payload = DOC + f"\n\n<!-- run {uuid.uuid4().hex} -->\n"
        files = {"file": (TEST_FILENAME, io.BytesIO(payload.encode("utf-8")), "text/markdown")}
        r = c.post(f"{BACKEND}/documents/upload", files=files)
        check("upload", r.status_code == 200, r.json().get("message", ""))
        if r.status_code != 200:
            return 1

        # ---------- 3. 轮询索引状态 ----------
        print("\n[3] 等待后台索引完成")
        deadline = time.time() + 180
        status = None
        while time.time() < deadline:
            lst = c.get(f"{BACKEND}/documents", params={"size": 50}).json()
            items = lst.get("items", [])
            # 按文件名精确匹配，避免误取知识库里已有的其它文档
            target = next((i for i in items if i["filename"] == TEST_FILENAME), None)
            if target:
                doc_id = target["id"]
                status = target["status"]
                if status in ("ready", "failed"):
                    break
            time.sleep(2)

        check("索引完成", status == "ready", f"status={status} doc_id={doc_id}")
        if status != "ready":
            if doc_id:
                info = c.get(f"{BACKEND}/documents/{doc_id}").json()
                print("    错误信息:", info.get("error_msg"))
            return 1

        # ---------- 4. 列表与详情 ----------
        print("\n[4] 列表与正文")
        r = c.get(f"{BACKEND}/documents", params={"page": 1, "size": 10})
        check(
            "documents 列表",
            r.status_code == 200 and r.json()["total"] >= 1,
            f"total={r.json()['total']}",
        )

        r = c.get(f"{BACKEND}/documents/{doc_id}")
        body = r.json()
        check(
            "文档详情",
            r.status_code == 200 and bool(body.get("summary")),
            f"摘要={len(body.get('summary') or '')}字 分块={body.get('chunk_count')}",
        )
        check(
            "保留原始文件名",
            body.get("filename") == TEST_FILENAME,
            f"filename={body.get('filename')}",
        )
        chunk_count = body.get("chunk_count") or 0
        vectors_after_upload = c.get(f"{BACKEND}/stats").json().get("total_vectors", 0)

        r = c.get(f"{BACKEND}/documents/{doc_id}/content")
        check(
            "正文预览",
            r.status_code == 200 and "BGE-M3" in r.json().get("content", ""),
            f"{len(r.json().get('content', ''))} 字符",
        )

        # ---------- 5. 检索 ----------
        print("\n[5] 语义检索")
        r = c.post(f"{BACKEND}/search", json={"query": "用了什么嵌入模型？", "top_k": 3})
        data = r.json()
        check(
            "search",
            r.status_code == 200 and data["total_chunks"] > 0,
            f"命中片段={data['total_chunks']} 文档={len(data['documents'])}",
        )
        if data["documents"]:
            print(f"    最高分: {data['documents'][0]['best_score']}")

        # ---------- 6. 问答 ----------
        print("\n[6] RAG 问答（一次性）")
        r = c.post(
            f"{BACKEND}/qa",
            json={
                "question": "向量数据库用的是什么索引？",
                "top_k": 3,
                # 测试不写历史对话，否则侧边栏会被测试数据塞满
                "save_history": False,
                "use_cache": False,
            },
        )
        data = r.json()
        check(
            "qa",
            r.status_code == 200 and bool(data.get("answer")),
            f"引用={len(data.get('citations', []))}条",
        )
        print("    回答:", (data.get("answer") or "")[:160].replace("\n", " "))

        # ---------- 7. 流式问答（SSE） ----------
        print("\n[7] 流式问答（SSE）")
        events = {"meta": 0, "delta": 0, "done": 0}
        text_parts = []
        with c.stream(
            "POST",
            f"{BACKEND}/qa/stream",
            json={
                "question": "下周计划做什么？",
                "top_k": 3,
                "save_history": False,
                "use_cache": False,
            },
        ) as resp:
            current = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    current = line[6:].strip()
                    if current in events:
                        events[current] += 1
                elif line.startswith("data:") and current == "delta":
                    try:
                        text_parts.append(json.loads(line[5:].strip()).get("text", ""))
                    except json.JSONDecodeError:
                        pass
        answer = "".join(text_parts)
        check(
            "qa/stream",
            events["meta"] >= 1 and events["delta"] >= 1 and events["done"] >= 1,
            f"meta={events['meta']} delta={events['delta']} done={events['done']}",
        )
        check("流式答案非空", len(answer) > 10, f"{len(answer)} 字")
        print("    回答:", answer[:160].replace("\n", " "))

        # ---------- 8. 知识图谱 ----------
        print("\n[8] 知识图谱")
        r = c.get(f"{BACKEND}/graph", params={"min_score": 0.45})
        data = r.json()
        check(
            "graph",
            r.status_code == 200 and data["count"] >= 1,
            f"节点={data['count']} 边={len(data['edges'])} 最高分={data.get('max_score')}",
        )

        # ---------- 9. Vite 代理 ----------
        print("\n[9] Vite 代理连通性")
        try:
            r = httpx.get(f"{PROXY}/stats", timeout=15.0)
            check("前端代理 /api", r.status_code == 200,
                  f"documents={r.json().get('total_documents')}")
        except Exception as exc:
            check("前端代理 /api", False, str(exc)[:100])

        # ---------- 10. 删除 ----------
        print("\n[10] 清理测试文档")
        r = c.delete(f"{BACKEND}/documents/{doc_id}")
        check("delete", r.status_code == 200, r.json().get("message", ""))

        # Milvus 的 count(*) 有秒级一致性延迟，轮询等计数回落
        target = vectors_after_upload - chunk_count
        vectors_after = None
        deadline = time.time() + 25
        while time.time() < deadline:
            vectors_after = c.get(f"{BACKEND}/stats").json().get("total_vectors")
            if vectors_after is not None and vectors_after <= target:
                break
            time.sleep(1.5)
        check(
            "删除后向量数减少",
            vectors_after is not None and vectors_after <= target,
            f"{vectors_after_upload} -> {vectors_after}（本应减少 {chunk_count} 条）",
        )

    # ---------- 汇总 ----------
    print("\n" + "=" * 56)
    print(f"  通过 {len(passed)} 项 / 失败 {len(failed)} 项")
    if failed:
        print("  失败项:", ", ".join(failed))
    print("=" * 56)
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n测试异常: {exc}")
        sys.exit(1)
