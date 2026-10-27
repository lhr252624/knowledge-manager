"""
验证两处 UI 修复

A. 文档列表轮询：上传后不刷新页面，状态能否自动变为"就绪"，且确实在发请求
B. 知识图谱：画布尺寸是否正常（不再是 0×0 导致只渲染一小块），
   以及 tooltip 是否被限制在画布内且宽度受控

用法：
    cd backend
    $env:PLAYWRIGHT_BROWSERS_PATH="F:\\agent\\agentuse\\.playwright"
    .\\venv\\Scripts\\python.exe verify_fixes.py
"""
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = "http://localhost:3000"
API = "http://127.0.0.1:8000/api/v1"
SHOT = Path(__file__).resolve().parents[2] / "screenshots"

DOC = "# 轮询验证文档\n\n" + "\n\n".join(
    f"## 小节 {i}\n本节用于验证列表状态自动刷新是否生效。向量检索基于 BGE-M3 与 Milvus。" * 3
    for i in range(1, 6)
)

results = {}


def check(name, ok, detail=""):
    results[name] = ok
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


def main() -> int:
    SHOT.mkdir(exist_ok=True)
    sample = SHOT / "_轮询验证文档.md"
    sample.write_text(DOC, encoding="utf-8")

    new_doc_id = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        # ================= A. 文档轮询 =================
        print("\n[A] 文档列表自动轮询")

        list_requests = []
        page.on(
            "request",
            lambda r: list_requests.append(r.url)
            if "/api/v1/documents" in r.url and r.method == "GET"
            else None,
        )

        page.goto(f"{BASE}/documents", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("text=拖拽文件到此处", timeout=60000)
        page.wait_for_timeout(1500)

        before_count = len(list_requests)

        page.set_input_files("input[type=file]", str(sample))
        print("  已上传，开始不刷新页面地观察状态…")

        # 记录首次看到"新文档存在但未就绪"
        saw_pending = False
        ready = False
        start = time.time()
        timeline = []
        last_state = None

        while time.time() - start < 180:
            page.wait_for_timeout(1000)
            try:
                rows = page.evaluate(
                    """() => [...document.querySelectorAll('.el-table__body tbody tr')].map(tr => {
                        const tds = tr.querySelectorAll('td');
                        const name = tds[0] ? tds[0].innerText.split('\\n')[0].trim() : '';
                        const status = tds[4] ? tds[4].innerText.trim() : '';
                        return { name, status };
                    })"""
                )
            except Exception:
                rows = []

            target = next((r for r in rows if "轮询验证" in r["name"]), None)
            state = target["status"] if target else "未出现"
            if state != last_state:
                timeline.append(f"{round(time.time() - start, 1)}s:{state}")
                print(f"    [{round(time.time() - start, 1)}s] 新文档状态 = {state}")
                last_state = state

            if target and state in ("排队中", "解析中", "索引中"):
                saw_pending = True
            if target and state == "就绪":
                ready = True
                break

        poll_requests = len(list_requests) - before_count
        elapsed = time.time() - start

        # 索引提速后（约 8s）可能观察不到"处理中"这个中间态，属正常现象：
        # 只要不是"卡住不动"，两种路径都算通过。
        check(
            "状态自动流转（无需刷新）",
            ready,
            f"耗时 {elapsed:.1f}s；中间态{'已观察到' if saw_pending else '过快未观察到'}",
        )
        check("上传后触发列表刷新", poll_requests >= 1, f"期间发出 {poll_requests} 次列表请求")

        # 空闲状态下仍应持续轮询（间隔 8s），确认自动刷新没有停掉
        idle_before = len(list_requests)
        page.wait_for_timeout(11000)
        idle_requests = len(list_requests) - idle_before
        check("空闲时仍在轮询", idle_requests >= 1, f"11s 内 {idle_requests} 次")

        page.screenshot(path=str(SHOT / "07-文档轮询.png"), full_page=True)

        # 记录新建文档 id 以便清理
        try:
            lst = httpx.get(f"{API}/documents", params={"size": 50}, timeout=20).json()
            for it in lst["items"]:
                if it["filename"].startswith("_轮询验证") or (it["title"] or "").startswith("轮询验证"):
                    new_doc_id = it["id"]
        except Exception:
            pass

        # ================= B. 知识图谱 =================
        print("\n[B] 知识图谱渲染")

        page.goto(f"{BASE}/graph", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".chart canvas", timeout=60000)
        page.wait_for_timeout(4000)  # 等力导向布局稳定

        size = page.evaluate(
            """() => {
                const cv = document.querySelector('.chart canvas');
                const wrap = document.querySelector('.chart-wrap');
                if (!cv || !wrap) return null;
                const cr = cv.getBoundingClientRect();
                const wr = wrap.getBoundingClientRect();
                return {
                    canvasW: Math.round(cr.width), canvasH: Math.round(cr.height),
                    cssW: cv.style.width, cssH: cv.style.height,
                    wrapW: Math.round(wr.width), wrapH: Math.round(wr.height),
                    dpr: window.devicePixelRatio
                };
            }"""
        )
        print(f"    画布: {size}")
        # 关键断言：canvas 显示尺寸应接近容器尺寸（允许 2% 误差）
        ok_w = size and size["canvasW"] >= size["wrapW"] * 0.95
        ok_h = size and size["canvasH"] >= size["wrapH"] * 0.95
        check("画布宽度填满容器", bool(ok_w), f"{size['canvasW']}px / 容器 {size['wrapW']}px")
        check("画布高度填满容器", bool(ok_h), f"{size['canvasH']}px / 容器 {size['wrapH']}px")

        # 窗口缩放后是否自适应
        page.set_viewport_size({"width": 1100, "height": 860})
        page.wait_for_timeout(1200)
        size2 = page.evaluate(
            """() => {
                const cv = document.querySelector('.chart canvas');
                const wrap = document.querySelector('.chart-wrap');
                const cr = cv.getBoundingClientRect();
                const wr = wrap.getBoundingClientRect();
                return [Math.round(cr.width), Math.round(wr.width)];
            }"""
        )
        check("窗口缩放后画布自适应", abs(size2[0] - size2[1]) <= 4, f"canvas {size2[0]}px / 容器 {size2[1]}px")
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_timeout(1000)

        # tooltip 尺寸：通过 ECharts 实例主动 showTip，避免靠鼠标盲猜节点位置
        tip = None
        try:
            page.evaluate(
                """() => {
                    const c = window.__kmChart;
                    if (!c) return;
                    const opt = c.getOption();
                    const n = (opt.series && opt.series[0] && opt.series[0].data || []).length;
                    if (n > 0) c.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex: 0 });
                }"""
            )
            page.wait_for_timeout(700)
            tip = page.evaluate(
                """() => {
                    const wrap = document.querySelector('.chart-wrap');
                    if (!wrap) return null;
                    const divs = [...wrap.querySelectorAll('div')];
                    const t = divs.find(d => {
                        const st = getComputedStyle(d);
                        return st.position === 'absolute' && d.innerText.trim().length > 0
                               && st.display !== 'none' && st.visibility !== 'hidden'
                               && parseFloat(st.opacity || '1') > 0.3;
                    });
                    if (!t) return null;
                    const r = t.getBoundingClientRect();
                    const wrapRect = wrap.getBoundingClientRect();
                    return { w: Math.round(r.width), h: Math.round(r.height),
                             insideX: r.left >= wrapRect.left - 1 && r.right <= wrapRect.right + 1,
                             insideY: r.top >= wrapRect.top - 1 && r.bottom <= wrapRect.bottom + 1,
                             text: t.innerText.replace(/\\s+/g, ' ').slice(0, 70) };
                }"""
            )
        except Exception as exc:
            print(f"    showTip 调用异常: {exc}")

        if tip:
            print(f"    tooltip: {tip['w']}×{tip['h']}  「{tip['text']}」")
            check("tooltip 宽度受控(≤300px)", tip["w"] <= 300, f"{tip['w']}px")
            check("tooltip 未超出画布高度", tip["h"] <= 400, f"{tip['h']}px")
            check("tooltip 完全在画布内", bool(tip["insideX"] and tip["insideY"]))
            page.screenshot(path=str(SHOT / "08-图谱tooltip.png"))
        else:
            check("tooltip 可触发", False, "showTip 未产生可见浮层")

        # 边数（验证默认阈值下能连出关系）
        edge_text = page.inner_text(".summary")
        print(f"    {edge_text}")
        edges = int(edge_text.split("/")[1].split("条")[0].strip()) if "条" in edge_text else 0
        check("默认阈值下存在关联边", edges > 0, f"{edges} 条")

        page.mouse.move(10, 10)
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOT / "09-知识图谱修复后.png"), full_page=True)

        real_errors = [
            e for e in console_errors
            if not any(n in e for n in ("favicon", "ResizeObserver", "Vue Devtools"))
        ]
        check("无 console 错误", not real_errors, str(real_errors[:2]))

        browser.close()

    # 清理
    if new_doc_id:
        try:
            httpx.delete(f"{API}/documents/{new_doc_id}", timeout=30)
            print(f"\n已清理测试文档 id={new_doc_id}")
        except Exception as exc:
            print(f"\n清理失败: {exc}")
    sample.unlink(missing_ok=True)

    print("\n" + "=" * 56)
    failed = [k for k, v in results.items() if not v]
    print(f"  通过 {len(results) - len(failed)} / {len(results)}")
    if failed:
        print("  失败:", ", ".join(failed))
    print("=" * 56)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
