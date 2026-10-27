"""索引耗时测量：分别测冷启动（模型未加载）与热状态（模型已在显存）"""
import io
import time

import httpx

API = "http://127.0.0.1:8000/api/v1"

DOC = """# 时间测量文档

## 一、背景
本文用于测量索引流水线耗时，覆盖解析、分块、向量化与摘要四个阶段。

## 二、方法
每篇文档上传后立即返回，索引在后台线程执行。客户端轮询文档状态，
当状态由 pending/parsing/indexing 变为 ready 时记录总耗时。

## 三、结果
冷启动时需额外加载 BGE-M3 权重（约 2.2GB），耗时会明显高于热状态。
热状态下耗时主要来自摘要阶段调用大模型。

## 四、结论
将摘要输入从 6000 字符缩短到 2500 字符，并把输出上限限制为 400 tokens，
可显著降低单篇文档的端到端耗时。
""" * 2


def run_once(label: str) -> float:
    # 加随机标记，避免被文档哈希去重（否则第二次上传会被直接跳过）
    import uuid

    body = DOC + f"\n\n<!-- 测量批次 {uuid.uuid4().hex[:8]} -->\n"
    files = {"file": (f"{label}.md", io.BytesIO(body.encode("utf-8")), "text/markdown")}
    start = time.time()
    r = httpx.post(f"{API}/documents/upload", files=files, timeout=60)
    r.raise_for_status()

    doc_id = None
    deadline = time.time() + 300
    while time.time() < deadline:
        lst = httpx.get(f"{API}/documents", params={"size": 10}, timeout=30).json()
        target = next(
            (i for i in lst["items"] if label in (i["filename"] or "") or label in (i["title"] or "")),
            None,
        )
        if target:
            doc_id = target["id"]
            if target["status"] == "ready":
                elapsed = time.time() - start
                summary_len = len(target.get("summary") or "")
                print(f"  {label}: 就绪，耗时 {elapsed:.1f}s，摘要 {summary_len} 字")
                httpx.delete(f"{API}/documents/{doc_id}", timeout=60)
                return elapsed
            if target["status"] == "failed":
                print(f"  {label}: 失败 -> {target.get('error_msg')}")
                httpx.delete(f"{API}/documents/{doc_id}", timeout=60)
                return -1
        time.sleep(1.5)

    print(f"  {label}: 超时")
    if doc_id:
        httpx.delete(f"{API}/documents/{doc_id}", timeout=60)
    return -1


if __name__ == "__main__":
    print("索引耗时测量")
    cold = run_once("耗时测量A")
    warm = run_once("耗时测量B")
    if cold > 0 and warm > 0:
        print(f"\n冷启动 {cold:.1f}s  ->  热状态 {warm:.1f}s")
