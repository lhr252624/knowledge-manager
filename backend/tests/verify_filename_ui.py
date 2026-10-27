"""
文件名匹配的 UI 验证

检查几个页面是否真的把"文件名"用起来了：
    检索页  名称命中提示 / 文档卡的"文件名匹配"标签
    问答页  引用卡片里显示文件名
    图谱页  节点 tooltip 带文件名、文件名关联开关、虚线边
    文档页  改名入口

测试数据由脚本自己创建：上传一篇文件名唯一、内容唯一的样例文档，
跑完再删掉。因此不依赖知识库里已有的任何文档，换台机器也能直接跑。

用法：
    cd backend
    $env:TMP="$PWD\..\.tmp"; $env:TEMP="$env:TMP"
    $env:PLAYWRIGHT_BROWSERS_PATH="$PWD\..\.playwright"
    .\\venv\\Scripts\\python.exe tests\\verify_filename_ui.py
"""
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = "http://localhost:3000"
API = "http://127.0.0.1:8000/api/v1"
SHOT = Path(__file__).resolve().parents[2] / "screenshots"

# 文件名里带随机后缀：既保证唯一（索引按内容 hash 去重），
# 又让"按名字检索"这件事有明确的正确答案
SAMPLE_STEM = f"文件名验证-{uuid.uuid4().hex[:8]}"
SAMPLE_NAME = f"{SAMPLE_STEM}.md"

# 结论里放一个不常见的数字，方便断言引用确实来自这篇文档
SAMPLE = """# 文件名验证样例文档

## 结论
本次验证的冲煮水温是 92 度，粉水比 1:15，闷蒸时间 30 秒。

## 说明
这份文档由 verify_filename_ui.py 自动生成，用来验证"按文件名检索"和
"问答引用里带文件名"两条链路是否正常工作。跑完会自动删除。
"""

passed: list[str] = []
failed: list[str] = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


def upload_sample() -> int:
    """上传样例文档并等待索引就绪，返回 document_id"""
    path = Path(tempfile.gettempdir()) / SAMPLE_NAME
    path.write_text(SAMPLE + f"\n<!-- {uuid.uuid4().hex} -->\n", encoding="utf-8")
    with httpx.Client(timeout=120) as client:
        files = {"file": (SAMPLE_NAME, path.read_bytes(), "text/markdown")}
        resp = client.post(f"{API}/documents/upload", params={"generate_summary": "false"}, files=files)
        resp.raise_for_status()

        deadline = time.time() + 180
        while time.time() < deadline:
            items = client.get(f"{API}/documents", params={"page": 1, "size": 100}).json()["items"]
            for item in items:
                if item["filename"] == SAMPLE_NAME and item["status"] == "ready":
                    print(f"    已就绪 id={item['id']}，分块 {item['chunk_count']}")
                    return item["id"]
            time.sleep(2)
    raise RuntimeError("样例文档索引超时")


def conversation_ids() -> set:
    """当前所有会话 id（跑完只删自己新建的）"""
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API}/conversations", params={"limit": 500})
        return {c["id"] for c in resp.json().get("items", [])}


def cleanup(conv_before) -> None:
    """删掉本次上传的样例文档和本次新建的会话（不留下任何测试数据）"""
    try:
        with httpx.Client(timeout=120) as client:
            items = client.get(f"{API}/documents", params={"page": 1, "size": 100}).json()["items"]
            for item in items:
                if item["filename"] == SAMPLE_NAME:
                    client.delete(f"{API}/documents/{item['id']}")
                    print(f"    已清理测试文档 id={item['id']}")
            # 问答页提问会真的建会话，不清掉会留在用户侧边栏
            for cid in conversation_ids() - conv_before:
                client.delete(f"{API}/conversations/{cid}")
                print(f"    已清理测试会话 id={cid}")
    except Exception as exc:
        print(f"    清理失败: {exc}")


def main() -> int:
    SHOT.mkdir(exist_ok=True)
    console_errors: list[str] = []

    print(f"\n[0] 准备样例文档 {SAMPLE_NAME}")
    conv_before = conversation_ids()
    try:
        doc_id = upload_sample()
    except Exception as exc:
        print(f"  [FAIL] 样例文档上传失败：{exc}")
        print("  提示：确认后端 http://127.0.0.1:8000 在跑，且嵌入模型已就绪")
        # 上传可能已经在服务端建好了文档（只是客户端连接被重置），照样清一遍
        print("\n[5] 清理")
        cleanup(conv_before)
        return 1
    check("样例文档已索引", bool(doc_id), f"id={doc_id}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on(
                "console",
                lambda m: console_errors.append(m.text) if m.type == "error" else None,
            )

            # ================= 1. 检索页 =================
            print("\n[1] 语义检索页")
            page.goto(f"{BASE}/search", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("input[type=text]", timeout=60000)
            page.wait_for_timeout(800)

            check("有「文件名匹配」开关", page.locator("text=文件名匹配").count() > 0)

            # 只输入文件名（不带 .md），验证"按文档名检索"这条通道
            page.fill("input[type=text]", SAMPLE_STEM)
            page.click("button:has-text('检索')")
            page.wait_for_selector(".name-hit", timeout=60000)
            page.wait_for_timeout(600)

            alert_text = page.locator(".name-hit").inner_text()
            check(
                "名称命中提示出现",
                "文件名命中" in alert_text and SAMPLE_STEM in alert_text,
                alert_text.replace("\n", " ")[:80],
            )

            first_card = page.locator(".doc-card").first.inner_text()
            check("首位结果是该文档", SAMPLE_STEM in first_card, first_card.replace("\n", " ")[:60])
            check(
                "卡片显示文件名",
                page.locator(".doc-file").count() > 0,
                page.locator(".doc-file").first.inner_text(),
            )
            check(
                "卡片有「文件名命中」标签",
                page.locator(".doc-meta .el-tag:has-text('文件名命中')").count() > 0,
            )
            page.screenshot(path=str(SHOT / "10-检索-文件名命中.png"))

            # ================= 2. 问答页 =================
            print("\n[2] 智能问答页")
            page.goto(f"{BASE}/chat", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("textarea", timeout=60000)
            page.fill("textarea", f"{SAMPLE_STEM} 这份文档的结论是什么？")
            page.click("button:has-text('发送')")
            page.wait_for_selector(".citation", timeout=180000)
            page.wait_for_timeout(500)

            cites = page.locator(".citation").all_inner_texts()
            check("引用出现", len(cites) > 0, f"{len(cites)} 条")
            check(
                "引用里带文件名",
                any(SAMPLE_NAME in c for c in cites),
                " | ".join(c.replace("\n", " ") for c in cites)[:100],
            )
            check(
                "引用有「文件名命中」标签",
                page.locator(".citation .el-tag:has-text('文件名命中')").count() > 0,
            )
            page.screenshot(path=str(SHOT / "11-问答-引用带文件名.png"))

            # ================= 3. 图谱页 =================
            print("\n[3] 知识图谱页")
            graph_requests: list[str] = []
            page.on(
                "request",
                lambda r: graph_requests.append(r.url) if "/api/v1/graph" in r.url else None,
            )
            page.goto(f"{BASE}/graph", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector(".chart canvas", timeout=60000)
            page.wait_for_timeout(2500)

            check(
                "页面说明提到虚线=文件名关联",
                "虚线" in page.locator(".page-desc").first.inner_text(),
                page.locator(".page-desc").first.inner_text()[:60],
            )
            check("有「文件名关联」开关", page.locator("text=文件名关联").count() > 0)
            check(
                "图谱请求带 name_min_score",
                any("name_min_score" in u for u in graph_requests),
                f"{len(graph_requests)} 次请求",
            )

            info = page.evaluate(
                """() => {
                    const inst = window.__kmChart;
                    if (!inst) return { error: 'no chart instance' };
                    const opt = inst.getOption();
                    const fmt = opt.tooltip && opt.tooltip[0] && opt.tooltip[0].formatter;
                    const data = opt.series[0].data || [];
                    const links = opt.series[0].links || [];
                    const withFile = data.filter(n => n.filename).length;
                    let nodeHtml = '';
                    let edgeHtml = '';
                    try { nodeHtml = typeof fmt === 'function' ? fmt({dataType:'node', data: data[0]}) : ''; } catch (e) { nodeHtml = 'ERR ' + e.message; }
                    try {
                      const dashed = links.find(l => l.relation === 'filename') || links[0];
                      edgeHtml = dashed && typeof fmt === 'function' ? fmt({dataType:'edge', data: dashed}) : '';
                    } catch (e) { edgeHtml = 'ERR ' + e.message; }
                    return {
                      nodes: data.length,
                      links: links.length,
                      withFile: withFile,
                      nodeHtml: nodeHtml,
                      edgeHtml: edgeHtml
                    };
                }"""
            )
            check(
                "图表实例可用",
                not info.get("error") and info.get("nodes", 0) >= 1,
                json.dumps(info, ensure_ascii=False)[:120],
            )
            check(
                "每个节点都带 filename",
                info.get("withFile") == info.get("nodes") and info.get("nodes", 0) >= 1,
                f"{info.get('withFile')}/{info.get('nodes')}",
            )
            check(
                "节点 tooltip 显示文件名",
                "文件：" in (info.get("nodeHtml") or ""),
                (info.get("nodeHtml") or "").replace("<br>", " ")[:110],
            )
            check(
                "边 tooltip 说明关联类型",
                "语义相似度" in (info.get("edgeHtml") or ""),
                (info.get("edgeHtml") or "")[:80],
            )
            page.screenshot(path=str(SHOT / "12-图谱-文件名关联.png"))

            # ================= 4. 文档页 =================
            print("\n[4] 文档管理页")
            page.goto(f"{BASE}/documents", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("text=拖拽文件到此处", timeout=60000)
            page.wait_for_timeout(1500)

            row = page.locator(".el-table__row", has_text=SAMPLE_STEM).first
            check("列表里能找到该文档", row.count() > 0)
            page.screenshot(path=str(SHOT / "13-文档-改名弹窗.png"))

            # 只打开弹窗看文案，不真的保存，避免改动库里数据
            row.locator("button:has-text('改名')").click()
            page.wait_for_selector(".el-dialog", timeout=20000)
            page.wait_for_timeout(400)
            dialog = page.locator(".el-dialog").inner_text()
            check(
                "改名弹窗说明会重算向量",
                "重算" in dialog and "文件名" in dialog,
                dialog.replace("\n", " ")[:90],
            )
            page.click(".el-dialog button:has-text('取消')")
            page.wait_for_timeout(300)

            check("无 console 错误", not console_errors, str(console_errors[:2]))
            browser.close()
    finally:
        print("\n[5] 清理")
        cleanup(conv_before)

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
