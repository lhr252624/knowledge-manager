"""
嵌入服务双提供方验证（离线，不连网络、不需要 API Key）

要证明的事：
    A. 本地模式默认行为不变，且读取 model_tag / space_tag 不会触发模型加载
    B. 远程模式按百炼上限切片（10 条/次），25 条 = 3 次请求
    C. 返回顺序按 index 还原，与输入一一对应
    D. 远程返回的向量被 L2 归一化（否则 Milvus 的 IP 度量会失去余弦语义）
    E. 维度与配置不一致时直接报错，不会把脏向量写进库
    F. 429 会重试、401 不重试；缺 Key 时给出可执行的提示
    G. dimensions 参数只对 v3/v4 发送（v1/v2 传了会 400）

用法（纯离线，无需启动后端）：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\verify_embedding_provider.py

加 --live 会额外用真实配置请求一次百炼接口（需要已配置 DASHSCOPE_API_KEY）：
    .\\venv\\Scripts\\python.exe tests\\verify_embedding_provider.py --live
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services import embedding_service as es_mod  # noqa: E402

settings = get_settings()
svc = es_mod.embedding_service
real_post = httpx.post
real_sleep = __import__("time").sleep

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name} {detail}")


class FakeResp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = text or str(self._body)

    def json(self):
        return self._body


def make_body(inputs, dim, reverse=False):
    """首维用文本长度、其余维填 1，归一化后首维仍随长度单调，便于反查顺序"""
    data = [
        {
            "index": i,
            "object": "embedding",
            "embedding": [float(len(t))] + [1.0] * (dim - 1),
        }
        for i, t in enumerate(inputs)
    ]
    if reverse:
        data.reverse()
    return {"data": data, "model": "fake", "object": "list", "usage": {"total_tokens": 1}}


def expected_first(text, dim):
    """L2 归一化后首维的期望值"""
    n = len(text)
    return n / math.sqrt(n * n + (dim - 1))


def install(handler):
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return handler(len(calls), kwargs.get("json") or {})

    httpx.post = fake_post
    return calls


def use_remote(**overrides):
    settings.embedding_provider = "dashscope"
    settings.dashscope_api_key = "sk-test"
    settings.embedding_dim = 8
    settings.dashscope_embedding_model = "text-embedding-v3"
    settings.dashscope_batch_size = 10
    settings.dashscope_max_retries = 2
    for key, value in overrides.items():
        setattr(settings, key, value)


def restore():
    httpx.post = real_post
    __import__("time").sleep = real_sleep
    settings.embedding_provider = "local"
    settings.embedding_dim = 1024


def main() -> int:
    saved = {
        "embedding_provider": settings.embedding_provider,
        "embedding_dim": settings.embedding_dim,
        "dashscope_api_key": settings.dashscope_api_key,
        "dashscope_embedding_model": settings.dashscope_embedding_model,
        "dashscope_batch_size": settings.dashscope_batch_size,
        "dashscope_max_retries": settings.dashscope_max_retries,
    }

    try:
        # ---------- A. 本地模式 ----------
        print("\n[1] 本地模式元信息")
        settings.embedding_provider = "local"
        settings.embedding_dim = 1024
        check("is_remote 为 False", svc.is_remote is False)
        check("model_tag 不带 dashscope 前缀", svc.model_tag.startswith("local:"))
        check("space_tag 形如 local_bge_m3_1024", svc.space_tag == "local_bge_m3_1024", svc.space_tag)
        check("读取元信息未加载本地模型（_model 仍为 None）", svc._model is None)

        # ---------- B/C/D. 远程切片 + 顺序 + 归一化 ----------
        print("\n[2] 远程切片、顺序还原、L2 归一化")
        use_remote()
        calls = install(lambda n, body: FakeResp(200, make_body(body["input"], settings.embedding_dim, reverse=True)))

        texts = [f"第{i}段" * (i + 1) for i in range(25)]
        vectors = svc.encode(texts, normalize=True)

        check("25 条切成 3 次请求", len(calls) == 3, f"实际 {len(calls)} 次")
        sizes = [len(c["json"]["input"]) for c in calls]
        check("批次大小 10/10/5", sizes == [10, 10, 5], str(sizes))
        check("返回条数一致", len(vectors) == 25, f"实际 {len(vectors)}")
        check("维度正确", all(len(v) == settings.embedding_dim for v in vectors))

        # 倒序返回后仍按 index 还原：首维应等于该文本的期望归一化值
        ordered = all(
            abs(vectors[i][0] - expected_first(texts[i], settings.embedding_dim)) < 1e-6
            for i in range(25)
        )
        check("倒序响应按 index 还原顺序", ordered)

        norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
        check("全部 L2 归一化（norm≈1）", all(abs(n - 1.0) < 1e-5 for n in norms), f"样例 {norms[0]:.6f}")

        check("请求地址拼接正确", calls[0]["url"].endswith("/compatible-mode/v1/embeddings"), calls[0]["url"])
        check("带 Bearer 鉴权头", calls[0]["headers"].get("Authorization") == "Bearer sk-test")
        check("v3 发送 dimensions", calls[0]["json"].get("dimensions") == 8)

        # ---------- 单条输入 ----------
        print("\n[3] 单条字符串输入")
        install(lambda n, body: FakeResp(200, make_body(body["input"], settings.embedding_dim)))
        one = svc.encode("单条")
        check("单条返回一维列表", isinstance(one, list) and len(one) == settings.embedding_dim)

        # ---------- E. 维度不符 ----------
        print("\n[4] 维度校验")
        install(lambda n, body: FakeResp(200, make_body(body["input"], 16)))
        try:
            svc.encode(["维度不对"])
            check("维度不符应报错", False)
        except RuntimeError as exc:
            check("维度不符报错且提示改 EMBEDDING_DIM", "EMBEDDING_DIM" in str(exc), str(exc)[:80])

        # ---------- F. 重试策略 ----------
        print("\n[5] 重试策略")
        __import__("time").sleep = lambda _s: None
        state = {"n": 0}

        def flaky(n, body):
            state["n"] += 1
            if state["n"] == 1:
                return FakeResp(429, {"error": {"message": "rate limit"}}, "rate limit")
            return FakeResp(200, make_body(body["input"], settings.embedding_dim))

        install(flaky)
        vec = svc.encode("限流后成功")
        check("429 重试后成功", len(vec) == settings.embedding_dim and state["n"] == 2, f"请求 {state['n']} 次")

        calls_auth = install(lambda n, body: FakeResp(401, {"error": {"message": "bad key"}}, "bad key"))
        try:
            svc.encode(["鉴权失败"])
            check("401 应直接抛错", False)
        except RuntimeError as exc:
            check("401 不重试（仅 1 次请求）", len(calls_auth) == 1, f"实际 {len(calls_auth)} 次")
            check("401 提示检查 Key", "DASHSCOPE_API_KEY" in str(exc))

        # ---------- G. dimensions 只对 v3/v4 发送 ----------
        print("\n[6] dimensions 参数兼容性")
        use_remote(dashscope_embedding_model="text-embedding-v2")
        calls_v2 = install(lambda n, body: FakeResp(200, make_body(body["input"], settings.embedding_dim)))
        svc.encode(["v2 模型"])
        check("v2 不发送 dimensions", "dimensions" not in calls_v2[0]["json"])

        # ---------- 缺 Key ----------
        print("\n[7] 缺少 Key")
        use_remote(dashscope_api_key="")
        try:
            svc.encode(["没有 key"])
            check("缺 Key 应报错", False)
        except RuntimeError as exc:
            check("缺 Key 提示可执行信息", "DASHSCOPE_API_KEY" in str(exc))

        # ---------- 远程集合隔离 ----------
        print("\n[8] 向量空间隔离")
        use_remote()
        check("远程 space_tag 独立", svc.space_tag == "ds_text_embedding_v3_8", svc.space_tag)
        from app.services.milvus_store import milvus_store

        remote_name = milvus_store._collection_name()
        settings.embedding_provider = "local"
        local_name = milvus_store._collection_name()
        check("远程集合名自动加后缀", remote_name != local_name, f"{local_name} / {remote_name}")
        check("本地集合名保持原样", local_name == settings.milvus_collection, local_name)

    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        restore()

    if "--live" in sys.argv:
        print("\n[9] 真实接口冒烟（--live）")
        live_check()

    print(f"\n通过 {len(passed)} / {len(passed) + len(failed)}")
    if failed:
        print("失败：" + ", ".join(failed))
        return 1
    return 0


def live_check() -> bool:
    """用真实配置请求一次百炼，验证 Key / 端点 / 维度都正确"""
    key = settings.dashscope_api_key
    if not key:
        print("  [SKIP] 未配置 DASHSCOPE_API_KEY，跳过")
        return True

    saved_provider = settings.embedding_provider
    saved_dim = settings.embedding_dim
    try:
        settings.embedding_provider = "dashscope"
        settings.embedding_dim = 1024
        vectors = svc.encode(["知识库检索测试", "第二个句子"], normalize=True)
        check("真实接口返回 2 条", len(vectors) == 2)
        check("真实接口维度 1024", all(len(v) == 1024 for v in vectors))
        norm = math.sqrt(sum(x * x for x in vectors[0]))
        check("真实接口向量已归一化", abs(norm - 1.0) < 1e-5, f"norm={norm:.6f}")
        # 相同文本应得到相同向量，不同文本应不同
        same = svc.encode("知识库检索测试")
        check("同文本向量稳定", all(abs(a - b) < 1e-6 for a, b in zip(same, vectors[0])))
        check("不同文本向量不同", vectors[0][0] != vectors[1][0])
        return True
    except Exception as exc:
        print(f"  [FAIL] 真实接口调用失败：{exc}")
        return False
    finally:
        settings.embedding_provider = saved_provider
        settings.embedding_dim = saved_dim


if __name__ == "__main__":
    sys.exit(main())
