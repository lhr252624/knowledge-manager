"""
前端 UI 自动化测试（Playwright）

流程：
    概览页加载 -> 文档页上传文件 -> 等待索引就绪 -> 预览正文 -> 问答页流式提问
    -> 检索页检索 -> 图谱页渲染 -> 清理上传的样例文档
并收集浏览器 console 错误与页面异常。

用法：
    cd backend
    .\\venv\\Scripts\\python.exe ui_test.py
"""
import sys
import tempfile
import time
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:3000"
API = "http://127.0.0.1:8000/api/v1"
SHOT_DIR = Path(__file__).resolve().parents[2] / "screenshots"
SAMPLE_NAME = "样例文档-家庭年度预算与理财计划.md"

SAMPLE = """# 家庭年度预算与理财计划

## 收入情况
家庭税后月收入约 3.2 万元，其中工资收入 2.8 万元，理财收益约 4000 元。
年终奖预计 6 万元，全部计入储蓄。

## 支出结构
每月固定支出包括：房贷 9000 元、物业与水电 800 元、通勤 600 元。
弹性支出包括：餐饮 4000 元、孩子教育 3000 元、娱乐购物 2500 元。
合计月支出约 2 万元，月结余约 1.2 万元。

## 理财配置
备用金保留 6 个月支出约 12 万元，放在货币基金随取随用。
剩余资金按 4:4:2 配置于债券基金、宽基指数基金和高股息股票。
每季度末做一次再平衡，偏离目标比例超过 5 个百分点时调整。

## 风险控制
单一资产占比不超过总资产的 30%。不参与杠杆与场外配资。
每年复核一次保险配置，重疾险保额不低于 50 万元。
"""


def conversation_ids() -> set:
    """当前所有会话 id（用来跑完只删自己新建的，不动用户已有会话）"""
    import httpx

    resp = httpx.get(f"{API}/conversations", params={"limit": 500}, timeout=30)
    return {c["id"] for c in resp.json().get("items", [])}


def main() -> int:
    SHOT_DIR.mkdir(exist_ok=True)
    # 内容必须唯一：索引层按内容 hash 去重，如果和知识库里已有文档完全一致，
    # 新文档会被"已存在，跳过"直接吞掉 —— 页面上表现成"提示上传成功，但列表里没有这一行"
    # 落盘放到临时目录，别把 .md 混进截图目录里
    sample_path = Path(tempfile.gettempdir()) / SAMPLE_NAME
    marker = f"\n\n<!-- 测试运行标记 {uuid.uuid4().hex[:10]} -->\n"
    sample_path.write_text(SAMPLE + marker, encoding="utf-8")

    console_errors: list[str] = []
    page_errors: list[str] = []
    results: dict[str, bool] = {}
    conv_before = conversation_ids()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        # ---------- 1. 概览页 ----------
        print("[1] 概览页")
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector("text=文档总数", timeout=20000)
        results["概览页渲染"] = page.locator("text=多 Agent 驱动的本地知识库").count() > 0
        page.screenshot(path=str(SHOT_DIR / "01-概览.png"), full_page=True)
        print("    截图: 01-概览.png")

        # ---------- 2. 文档上传 ----------
        print("[2] 文档页 - 上传")
        page.goto(f"{BASE}/documents", wait_until="networkidle")
        page.wait_for_selector("text=拖拽文件到此处", timeout=20000)
        page.set_input_files("input[type=file]", str(sample_path))

        # 等待表格出现该文档并变为"就绪"
        ready = False
        deadline = time.time() + 180
        while time.time() < deadline:
            page.wait_for_timeout(2500)
            body = page.inner_text("body")
            if "样例文档" in body and "就绪" in body:
                ready = True
                break
        results["上传并索引就绪"] = ready
        page.screenshot(path=str(SHOT_DIR / "02-文档管理.png"), full_page=True)
        print(f"    截图: 02-文档管理.png (就绪={ready})")

        # ---------- 3. 文档预览 ----------
        print("[3] 文档预览抽屉")
        try:
            # 两个坑：
            #   1. body 里的"样例文档"可能只是上传进度条，表格行还没渲染出来，
            #      所以必须等 .el-table__row 真的出现
            #   2. 列表按更新时间排序，刚上传的不一定在第一行，
            #      早期版本直接点 .first 会打开别的文档（预览到的是用户简历）
            row = page.locator(".el-table__row", has_text="样例文档").first
            row.wait_for(state="visible", timeout=120000)
            row.get_by_role("button", name="查看").click()
            page.wait_for_selector(".el-drawer", timeout=15000)
            page.wait_for_timeout(1200)
            drawer_text = page.inner_text(".el-drawer")
            results["正文预览"] = "理财" in drawer_text or "预算" in drawer_text
            page.screenshot(path=str(SHOT_DIR / "03-文档预览.png"))
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)
        except Exception as exc:
            results["正文预览"] = False
            print("    预览失败:", exc)

        # ---------- 4. 智能问答 ----------
        print("[4] 问答页 - 流式提问")
        page.goto(f"{BASE}/chat", wait_until="domcontentloaded")
        page.wait_for_selector("textarea", timeout=20000)
        page.fill("textarea", "这个家庭的理财配置比例是怎样的？")
        page.get_by_role("button", name="发送").click()

        answered = False
        deadline = time.time() + 120
        while time.time() < deadline:
            page.wait_for_timeout(2000)
            body = page.inner_text("body")
            if "参考来源" in body or "备用金" in body or "货币基金" in body:
                answered = True
                page.wait_for_timeout(3000)  # 等流式输出稳定
                break
        results["问答有回答"] = answered
        results["问答带引用"] = "参考来源" in page.inner_text("body")
        page.screenshot(path=str(SHOT_DIR / "04-智能问答.png"), full_page=True)
        print(f"    截图: 04-智能问答.png (回答={answered})")

        # ---------- 5. 语义检索 ----------
        print("[5] 检索页")
        page.goto(f"{BASE}/search", wait_until="networkidle")
        page.wait_for_selector("input", timeout=20000)
        page.fill("input", "应急备用金保留多少")
        page.get_by_role("button", name="检索").click()

        searched = False
        deadline = time.time() + 90
        while time.time() < deadline:
            page.wait_for_timeout(2000)
            if "命中" in page.inner_text("body"):
                searched = True
                break
        results["检索有结果"] = searched
        page.screenshot(path=str(SHOT_DIR / "05-语义检索.png"), full_page=True)
        print(f"    截图: 05-语义检索.png (有结果={searched})")

        # ---------- 6. 知识图谱 ----------
        print("[6] 图谱页")
        page.goto(f"{BASE}/graph", wait_until="networkidle")
        page.wait_for_timeout(6000)
        canvas = page.locator("canvas")
        results["图谱canvas渲染"] = canvas.count() > 0
        page.screenshot(path=str(SHOT_DIR / "06-知识图谱.png"), full_page=True)
        print("    截图: 06-知识图谱.png")

        browser.close()

    # ---------- 清理 ----------
    # 上传会真的创建一篇文档，跑完必须删掉，否则每跑一次知识库就多一份
    try:
        import httpx

        items = (
            httpx.get(
                f"{API}/documents", params={"page": 1, "page_size": 100}, timeout=30
            )
            .json()
            .get("items", [])
        )
        for item in items:
            if item["filename"] == SAMPLE_NAME:
                httpx.delete(f"{API}/documents/{item['id']}", timeout=120)
                print(f"  已清理测试文档 id={item['id']}")
    except Exception as exc:
        print(f"  清理测试文档失败: {exc}")

    # 在问答页提问会真的建一条会话，不清掉就会留在用户侧边栏里
    try:
        import httpx

        for cid in conversation_ids() - conv_before:
            httpx.delete(f"{API}/conversations/{cid}", timeout=60)
            print(f"  已清理测试会话 id={cid}")
    except Exception as exc:
        print(f"  清理测试会话失败: {exc}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 56)
    for name, ok in results.items():
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}")
    print("=" * 56)

    # 过滤掉无关的第三方告警
    noise = ("favicon", "ResizeObserver", "Download the Vue Devtools")
    real_errors = [e for e in console_errors if not any(n in e for n in noise)]
    if real_errors:
        print("\n浏览器 console 错误:")
        for e in real_errors[:10]:
            print("  -", e[:200])
    else:
        print("\n浏览器 console: 无错误")

    if page_errors:
        print("\n页面异常:")
        for e in page_errors[:10]:
            print("  -", e[:200])
    else:
        print("页面异常: 无")

    ok_all = all(results.values()) and not real_errors and not page_errors
    print("\n结论:", "全部通过" if ok_all else "存在问题")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
