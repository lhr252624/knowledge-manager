# backend/tests

验证脚本。项目没有用 pytest，每个脚本自己打印 `[OK] / [FAIL]` 并返回退出码，
靠"最后一行通过数"判断。改动代码后按下表挑相关的跑一遍即可。

## 怎么运行

必须在 `backend/` 目录下执行（脚本会把 `backend/` 加进 `sys.path` 来 import `app`）：

```powershell
cd backend
.\venv\Scripts\python.exe tests\verify_filename.py
```

需要浏览器的脚本（Playwright）还要先设两个环境变量，否则会报
`EPERM ... mkdtemp` / 找不到浏览器：

```powershell
# 在 backend/ 目录执行（.. 即项目根目录），把临时目录放进仓库内
$root = (Resolve-Path "..").Path
$env:TMP="$root\.tmp"; $env:TEMP="$root\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH="$root\.playwright"
```

## 前置条件

| 标记 | 含义 |
|------|------|
| — | 不需要任何服务，纯算法/纯本地逻辑 |
| 后端 | 需要 `http://127.0.0.1:8000` 在跑 |
| 前端 | 需要 `http://localhost:3000` 在跑 |
| Playwright | 需要浏览器二进制 + 上面的环境变量 |
| 重排模型 | 会自己加载 `bge-reranker-large`（约 60 秒）；模型缺失时相关用例标记 SKIP 而不是失败 |

> **跑 UI 测试时不要同时改 `backend/` 下的文件。**
> 后端起的是 `uvicorn --reload`，默认盯着整个 `backend/`，改一下脚本就会触发热重启，
> 正在跑的浏览器会撞上半路重启，报一串 `500 (Internal Server Error)`。
> 启动时加 `--reload-dir app` 可以避免（README 里的启动命令已经带上）。

## 脚本清单

| 脚本 | 前置 | 覆盖内容 |
|------|------|----------|
| `smoke_test.py` | — | 端到端链路：造样例文档 → 索引入库 → 检索 → 问答 → 图谱 → 清理。不启动 HTTP 服务，直接调 Agent |
| `test_api.py` | 后端 | HTTP 接口：上传 / 轮询 / 检索 / 问答 / SSE 流式 / 图谱 / 删除，共 17 项 |
| `verify_qa_cache.py` | 后端 | 问答缓存：分词、守卫、未命中→命中→语序变体→数字变体、知识库变更后自动失效，共 36 项 |
| `verify_cache_rerank.py` | 后端 + 重排模型 | 三层命中（精确 / 词面 / 语义重排）、数字与否定词守卫、最小差异对必须拒绝、命中耗时，共 28 项 |
| `verify_graph_cache.py` | 后端 | 知识图谱的向量缓存：冷启动 / 复用 / 增量 / 删除级联，共 15 项 |
| `verify_filename.py` | 后端 | 文件名匹配：分词、名称相似度对称性、检索加分、图谱连边，共 45 项 |
| `verify_fixes.py` | 后端 + 前端 + Playwright | 轮询自动刷新、图谱画布尺寸、tooltip 不溢出，共 11 项 |
| `verify_history_ui.py` | 后端 + 前端 + Playwright | 侧栏会话列表、新对话、重命名、删除、刷新后仍在、缓存命中徽章，共 20 项 |
| `verify_filename_ui.py` | 后端 + 前端 + Playwright | 文件名在检索结果/知识图谱里的展示与改名弹窗（自建样例文档，跑完自动清理），共 19 项 |
| `ui_test.py` | 后端 + 前端 + Playwright | 全流程走查（概览/上传/问答/检索/图谱）并截图到 `screenshots/` |

## 约定

- 脚本必须能**重复运行**：自己造的数据自己清（删文档、清缓存、清会话），
  不要把测试文档留在知识库里
- UI 脚本在页面上提问会真的建会话：跑之前先记下已有会话 id，收尾只删**新增**的那些，
  不要动用户自己的会话（`ui_test.py` / `verify_filename_ui.py` 已经这么做）
- 往缓存里塞合成条目的脚本，收尾要清掉，否则界面上会看到 `答案0：…` 这种假答案
- 调用 `/qa` 时带 `save_history: false`，否则会污染用户的历史对话
- 结论写进脚本的 `check(...)` 文案里，失败时要能一眼看出是哪条规则被破坏了
- 截图统一输出到项目根的 `screenshots/`，不要写进 `backend/`
- **跑 UI 测试时不要同时改 `backend/` 下的文件**，后端热重启会让请求撞上连接重置

## 不在这里的东西

- `check_env.py`：环境自检（依赖、模型路径、数据库连通性）
- `migrate_filename.py`：老数据修复工具（把 UUID 占位名换回真实文件名）
- `measure_index.py`：索引耗时基准
- `evaluate_cache_matchers.py` / `evaluate_llm_judge.py`：缓存匹配算法的离线评测
  （59 个人工标注问题对，用来比较词面 / 语义 / 交叉编码器 / 大模型判定的召回与精确率）
