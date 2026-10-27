"""
历史对话 + 缓存命中的 UI 验证

检查仿 DeepSeek 的侧边栏是否真的可用：
    新对话 / 会话列表 / 点击切换 / 刷新后仍在 / 重命名 / 删除 / 缓存命中标记

用法：
    cd backend
    $env:PLAYWRIGHT_BROWSERS_PATH="F:\\agent\\agentuse\\.playwright"
    $env:TMP="F:\\agent\\agentuse\\.tmp"; $env:TEMP=$env:TMP
    .\\venv\\Scripts\\python.exe verify_history_ui.py
"""
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = "http://localhost:3000"
API = "http://127.0.0.1:8000/api/v1"
SHOT = Path(__file__).resolve().parents[2] / "screenshots"

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


QUESTION = "向量数据库用的是什么索引"


def main() -> int:
    SHOT.mkdir(exist_ok=True)
    created_ids = []
    console_errors = []

    # 先清空历史，避免旧会话干扰断言
    with httpx.Client(timeout=60.0) as c:
        try:
            c.delete(f"{API}/conversations")
        except Exception:  # noqa: BLE001
            pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )

        # ================= 1. 初始状态 =================
        print("\n[1] 侧边栏与空状态")
        page.goto(f"{BASE}/chat", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector(".history", timeout=60000)
        page.wait_for_timeout(1200)

        check("侧边栏存在", page.locator(".history").count() == 1)
        check("有「新对话」按钮", page.locator("button:has-text('新对话')").count() == 1)
        check(
            "空状态提示存在",
            "还没有历史对话" in page.locator(".history").inner_text(),
            page.locator(".history").inner_text().replace("\n", " ")[:40],
        )
        check("有缓存统计", "缓存" in page.locator(".cache-line").inner_text(),
              page.locator(".cache-line").inner_text().replace("\n", " "))
        page.screenshot(path=str(SHOT / "14-问答-历史侧边栏空状态.png"))

        # ================= 2. 第一次提问 =================
        print("\n[2] 第一次提问，应新建会话")
        page.fill("textarea", QUESTION)
        page.click("button:has-text('发送')")
        page.wait_for_selector(".msg.assistant .md", timeout=180000)
        page.wait_for_timeout(2500)

        url = page.url
        check("地址栏带上了会话 id", "/chat/" in url, url)
        conv_id = url.rstrip("/").split("/")[-1]
        if conv_id.isdigit():
            created_ids.append(int(conv_id))

        items = page.locator(".conv-item")
        check("侧边栏出现 1 个会话", items.count() == 1, f"{items.count()} 个")
        title = items.first.inner_text()
        check("会话标题取第一个问题", QUESTION[:6] in title, title.replace("\n", " ")[:40])

        # ================= 3. 重复提问应命中缓存 =================
        print("\n[3] 同一问题再问一次，应命中缓存")
        page.fill("textarea", QUESTION)
        page.click("button:has-text('发送')")
        page.wait_for_selector(".cache-badge", timeout=120000)
        page.wait_for_timeout(800)

        badges = page.locator(".cache-badge")
        check("出现缓存命中标记", badges.count() >= 1, f"{badges.count()} 个")
        badge_text = badges.first.inner_text()
        # 标记现在显示命中方式（完全相同 / 词面匹配 / 语义判定）
        check(
            "标记里带命中方式",
            "缓存命中" in badge_text
            and any(k in badge_text for k in ("完全相同", "词面匹配", "语义判定", "相似")),
            badge_text,
        )

        latency = page.locator(".latency")
        check(
            "显示耗时",
            latency.count() >= 1,
            f"{latency.first.inner_text() if latency.count() else '-'}",
        )
        page.screenshot(path=str(SHOT / "15-问答-缓存命中.png"))

        # ================= 4. 刷新后历史仍在 =================
        print("\n[4] 刷新页面后历史仍在")
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(".history", timeout=60000)
        page.wait_for_timeout(1800)

        items = page.locator(".conv-item")
        check("刷新后会话仍在", items.count() == 1, f"{items.count()} 个")
        msgs = page.locator(".msg")
        check("消息被恢复（4 条）", msgs.count() == 4, f"{msgs.count()} 条")
        check(
            "缓存标记也被恢复",
            page.locator(".cache-badge").count() >= 1,
            f"{page.locator('.cache-badge').count()} 个",
        )

        # ================= 5. 新对话 =================
        print("\n[5] 新对话")
        page.click("button:has-text('新对话')")
        page.wait_for_timeout(800)
        check("回到空状态", page.locator(".msg").count() == 0)
        check("地址回到 /chat", page.url.rstrip("/").endswith("/chat"), page.url)
        check("旧会话还在列表里", page.locator(".conv-item").count() == 1)

        # ================= 6. 重命名 =================
        print("\n[6] 重命名会话")
        page.wait_for_timeout(600)
        page.locator(".conv-item").first.hover()
        page.locator(".conv-item .conv-actions .el-icon").first.click()
        page.wait_for_selector(".el-message-box__input input", timeout=20000)
        page.fill(".el-message-box__input input", "改过名的会话")
        page.click(".el-message-box button:has-text('保存')")
        try:
            page.wait_for_function(
                "() => document.querySelector('.conv-item .conv-title')?.textContent?.trim() === '改过名的会话'",
                timeout=15000,
            )
        except Exception:  # noqa: BLE001
            pass
        titles = page.locator(".conv-item .conv-title").all_inner_texts()
        check("标题已更新", titles and titles[0] == "改过名的会话", str(titles))

        # ================= 7. 切换会话 =================
        print("\n[7] 点击会话切回历史")
        page.locator(".conv-item").first.click()
        page.wait_for_timeout(1500)
        check("恢复历史消息", page.locator(".msg").count() == 4, f"{page.locator('.msg').count()} 条")
        page.screenshot(path=str(SHOT / "16-问答-历史会话切换.png"))

        # ================= 8. 删除会话 =================
        print("\n[8] 删除会话")
        page.locator(".conv-item").first.hover()
        page.locator(".conv-item .conv-actions .el-icon").nth(1).click()
        page.wait_for_selector(".el-message-box", timeout=20000)
        page.click(".el-message-box button:has-text('确定')")
        page.wait_for_timeout(1500)
        check("列表已清空", page.locator(".conv-item").count() == 0,
              f"{page.locator('.conv-item').count()} 个")

        check("无 console 错误", not console_errors, str(console_errors[:2]))
        browser.close()

    # 兜底清理
    with httpx.Client(timeout=60.0) as c:
        for cid in created_ids:
            try:
                c.delete(f"{API}/conversations/{cid}")
            except Exception:  # noqa: BLE001
                pass

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
