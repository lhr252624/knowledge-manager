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
| `verify_summary_agent.py` | 仅 LLM（不启动后端） | 摘要 Agent 的因果验证：把结论埋到 2500 字之后（旧实现必漏），断言新实现轨迹覆盖尾部且摘要采纳结论；另含步数上限、重复读取防护、reduce 兜底、短文档走单次，共 16 项 |
| `verify_qa_agent.py` | LLM + Milvus | 闸门逻辑（确定性，直接测 `_needs_agent`）+ 集成：降级、轮次预算、强制循环、`run()` 同步路径，共 10 项 |
| `eval_qa_retrieval.py` | 后端 | 检索召回基线：12 条单轮 + 3 条多轮追问，按"召回片段里必须出现的关键串"判命中 |
| `eval_qa_agent.py` | 后端 | 答案级 A/B：同一批题跑 `/qa`，看关键串是否出现在答案或引用预览里，对比耗时与步数（题目写死在代码里，对应旧语料） |
| `eval_ab_qa.py` | 仅本地模型 + LLM（不启动后端） | 进程内 A/B：读 `ab_questions.json` 的 15 条标注题（题目出自当前库），翻 `qa_agent_enabled` 对比 agent 循环开/关的**答案命中 / 片段命中 / 文档命中**、耗时与步数；`--limit` 试水，`--only` 按 group 过滤 |

## 问答 A/B 的题目怎么出

`ab_questions.json` 每题长这样：

```json
{
  "id": "resume-recall",
  "group": "resume_agent",
  "type": "single | followup | refuse",
  "question": "ResumeAgent 在 200 多份简历上测出来的问题召回率是多少？",
  "key": {"document_id": 168, "chunk_index": 7, "string": "84.7%"},
  "followup": {"question": "那 MAE 呢", "key": {"document_id": 168, "chunk_index": 8, "string": "3.8"}},
  "note": "第四章 · 实测指标"
}
```

- `key.string` **必须逐字出现在该 `document_id` 的该 `chunk_index` 分块里**，否则"片段命中"永远是 0；
  提交前先跑 `--verify-annotations`（它直接查 Chunk 表，不加载模型），全部 `[OK]` 再跑正式评测
- `string` 取数字、阈值、指标名这类有辨识度的片段，别取"因此""但是"
- `string` 在答案里比对时**忽略空白**（"P95 延迟" 与 "P95延迟" 都算命中），但词序和数字仍然严格比
- 同一个答案的常见写法放 `also`（如 `"string": "项目深度（30%）"` 配 `"also": ["30%", "0.30"]`），
  任一命中即算答对——判的是"答案对不对"，不是"措辞像不像原文"
- `type: "refuse"` 的题不写 `key`，只判是否拒答；`type: "followup"` 的题评分只看追问那一轮，
  耗时按两轮之和计（用户实际等的时间）
- `B` 组的平均步数会明显小于 2：早退闸门（证据足够、形式类请求、不需要 agent）照样生效，
  这是线上真实行为；`A` 组因关掉 agent 会被内部标 `degraded`，统计时不把它算作异常
- 换语料只改这个 JSON，不动脚本

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
