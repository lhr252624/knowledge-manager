# 智能知识管理系统

> 把自己的文档变成一个能问答、能检索、能看关系的知识库 —— 全本地部署，除大模型外不依赖任何外部服务。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![Vue](https://img.shields.io/badge/Vue-3.x-42b883.svg)](https://vuejs.org/)

上传 PDF / Word / Markdown 之后，系统会自动解析、切块、向量化并入库。之后你可以用自然语言提问，
答案带原文引用；也可以看一张文档关系图，了解哪些文档在讲同一件事。

---

## 目录

- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
  - [方式一：Docker 一键起数据库（推荐）](#方式一docker-一键起数据库推荐)
  - [方式二：自己装 MySQL 与 Milvus](#方式二自己装-mysql-与-milvus)
  - [下载模型](#下载模型)
  - [安装后端](#安装后端)
  - [配置文件](#配置文件)
  - [启动服务](#启动服务)
- [配置项全表](#配置项全表)
- [主要接口](#主要接口)
- [测试](#测试)
- [目录结构](#目录结构)
- [设计说明：问答缓存的三层命中](#设计说明问答缓存的三层命中)
- [设计说明：摘要为什么是一个 agent](#设计说明摘要为什么是一个-agent)
- [设计说明：QA 为什么是 agent + 置信度闸门](#设计说明qa-为什么是-agent--置信度闸门)
- [常见问题](#常见问题)
- [安全提示与已知限制](#安全提示与已知限制)
- [服务器部署（可选）](#服务器部署可选)
- [开发约定](#开发约定)
- [License](#license)

---

## 核心特性

| 能力 | 说明 |
|------|------|
| **多格式入库** | PDF / DOCX / Markdown / TXT / CSV / LOG / JSON，单文件 ≤ 50MB，后台异步索引，前端可轮询进度 |
| **语义检索** | BGE-M3 生成 1024 维向量，Milvus HNSW + 内积检索；同时把文件名拼进向量，名字相关的文档也更容易召回 |
| **文件名独立通道** | 除了向量，还有一条按"文档名 / 标题"打分的通道；`作业2-王小明.pdf` 这类只在文件名里出现的信息不会丢 |
| **RAG 问答** | 检索到的分块 + 文件名一起喂给 DeepSeek，SSE 流式返回，答案下方列出引用来源与相似度 |
| **知识图谱** | 文档之间的语义相似边 + 文件名相似边（虚线），前端 ECharts 力导向图，可拖阈值滑块实时联动 |
| **问答缓存** | 同样的问题不必再问一遍模型：精确指纹 / 词面 BM25+余弦 / 语义召回+交叉编码器重排，三层命中，命中后**毫秒级**返回 |
| **会话历史** | DeepSeek 风格侧边栏，支持多会话、重命名、删除，追问时自动带上前几轮上下文 |
| **纯本地** | MySQL + Milvus + 本地嵌入模型都在自己机器上，只有调用大模型时才出网 |

实测（RTX 4060 Laptop 8GB）：同一个问题的第 2 次提问，从 **9063 ms 降到 73 ms**，答案完全一致。

---

## 技术栈

| 层 | 选型 |
|----|------|
| 前端 | Vue 3 + Vite + Element Plus + ECharts + axios |
| 后端 | FastAPI + Uvicorn + Pydantic v2 |
| 编排 | LangChain（LCEL 链式调用） |
| 大模型 | DeepSeek（`deepseek-chat`，OpenAI 兼容接口，可换成任何兼容实现） |
| 嵌入 | BGE-M3（sentence-transformers，1024 维，L2 归一化） |
| 重排 | bge-reranker-large（交叉编码器，可选，缺失自动降级） |
| 向量库 | Milvus 2.4（HNSW / IP） |
| 关系库 | MySQL 8（SQLAlchemy 2.0 同步 ORM） |
| 分词 | jieba + BM25（问答缓存的词面通道） |

---

## 系统架构

```
                    ┌──────────────────────────────────────────┐
   浏览器 ──────────▶│  Vue 3 前端（Vite :3000 / Nginx :80）      │
                    └───────────────┬──────────────────────────┘
                                    │  /api/v1   （dev 走 Vite 代理，prod 走 Nginx）
                    ┌───────────────▼──────────────────────────┐
                    │  FastAPI（:8000）                          │
                    │  documents / qa / graph / conversations   │
                    └───┬───────────┬──────────────┬───────────┘
                        │           │              │
          ┌─────────────▼──┐  ┌─────▼──────┐  ┌────▼─────────────┐
          │ 多 Agent       │  │ 问答缓存    │  │ MySQL 8          │
          │ IndexAgent     │  │ 三层命中    │  │ documents/chunks │
          │ SearchAgent    │  │            │  │ qa_cache         │
          │ QAAgent        │  │ 精确指纹    │  │ conversations    │
          │ SummaryAgent   │  │ 词面 BM25   │  │ doc_vectors      │
          │ RelationAgent  │  │ 语义+重排   │  └──────────────────┘
          └──┬──────────┬──┘  └────────────┘
             │          │
   ┌─────────▼───┐  ┌───▼─────────────────┐  ┌──────────────────┐
   │ BGE-M3      │  │ Milvus 2.4          │  │ DeepSeek API     │
   │ 本地嵌入     │  │ HNSW / IP / 1024d   │  │ （唯一的外部依赖） │
   └─────────────┘  └─────────────────────┘  └──────────────────┘
```

**Agent 分工**（`backend/app/agents/`，均继承 `BaseAgent`）：

| Agent | 职责 |
|-------|------|
| `IndexAgent` | 解析文档 → 按 hash 去重 → 切块 → 写 MySQL → 生成向量写 Milvus |
| `SearchAgent` | 向量召回 + 文件名通道加权，返回带分数的分块 |
| `QAAgent` | 检索 → 组装上下文 → 调 LLM → 流式输出；默认走 **ReAct 工具循环**，带置信度闸门（见[设计说明](#设计说明qa-为什么是-agent--置信度闸门)） |
| `SummaryAgent` | 摘要与关键词：短文档单次生成；长文档走 **ReAct 工具循环**（先看结构，再按需读原文） |
| `RelationAgent` | 文档相似度计算与知识图谱连边（名字边 + 语义边） |

**一次提问的完整链路**：

1. 前端 `POST /api/v1/qa/stream`
2. 先查问答缓存（精确 → 词面 → 语义重排），命中就直接返回，整个过程 < 150ms
3. 未命中：BGE-M3 编码问题 → Milvus 召回 top_k → 文件名通道加分
4. **置信度闸门**判断：单次检索够用就直答；不够（低分 / 短追问 / 通读对比类）才进 ReAct 工具循环
5. 走循环时，模型自己决定搜几次、改什么关键词、读哪几段，逐步累积证据
6. DeepSeek 流式生成 → SSE 逐块推给前端（`meta` / `step` / `delta` / `error` / `done`）
7. 顺手把「问题 + 答案」写进缓存，下次同样的问法直接命中

---

## 快速开始

### 前置条件

| 依赖 | 要求 | 说明 |
|------|------|------|
| Python | 3.10 ~ 3.12 | 3.13 部分依赖还没轮子 |
| Node.js | 18 或更高 | Vite 5 的要求 |
| Docker | 推荐，可选 | 用来一键起 MySQL + Milvus |
| 磁盘 | ≥ 6 GB | BGE-M3 约 2.2GB，重排模型约 2.2GB，依赖约 1.5GB |
| 显存 | 可选 | 没有显卡用 CPU 也能跑，只是慢一些 |
| DeepSeek API Key | 必须 | <https://platform.deepseek.com> 申请 |

```bash
git clone <你的仓库地址> knowledge-manager
cd knowledge-manager
```

> 下面命令以 Windows PowerShell 为主，Linux / macOS 只需把
> `backend\venv\Scripts\python.exe` 换成 `backend/venv/bin/python`。

### 方式一：Docker 一键起数据库（推荐）

只需要 MySQL 和 Milvus，后端和前端在宿主机直接跑（方便热重载）：

```bash
docker compose -f docker-compose.dev.yml up -d
```

起来的东西：

| 服务 | 宿主机端口 | 备注 |
|------|-----------|------|
| MySQL 8 | **3307** | 账号 `root`，密码 `root`，默认库 `knowledge_manager` |
| Milvus 2.4 | 19530 | 依赖下面的 etcd 与 MinIO |
| Milvus etcd | 内部 | 元数据 |
| MinIO | 9000 / 9001 | 对象存储，9001 是控制台 |

> 端口故意用 3307 而不是 3306，避免和你本机已有的 MySQL 打架。
> 如果你想让 MySQL 用 3306，改 `docker-compose.dev.yml` 里的 `"3307:3306"`，
> 同时把配置文件里的 `MYSQL_PORT` 改成对应的值。

### 方式二：自己装 MySQL 与 Milvus

- MySQL 8：建库的事不用管，后端启动时会 `CREATE DATABASE IF NOT EXISTS`
- Milvus 2.4：按[官方文档](https://milvus.io/docs/install_standalone-docker.md)起 standalone 即可，
  集合和索引由后端自动创建

装好后把配置文件里的 `MYSQL_HOST/PORT` 和 `MILVUS_HOST/PORT` 指过去。

### 下载模型

两个模型都放到项目根目录的 `models/` 下（该目录已在 `.gitignore` 里，不会提交）：

```bash
pip install -U huggingface_hub

# 国内网络建议先设镜像（PowerShell）
$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download BAAI/bge-m3            --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-large --local-dir models/bge-reranker-large
```

> 新版 `huggingface_hub` 的命令是 `hf download`，老版本是 `huggingface-cli download`，两个都行。
> 也可以从 [ModelScope](https://modelscope.cn/models/BAAI/bge-m3) 手动下载后解压到同一目录。

最终目录应该是这样：

```
models/
├── bge-m3/                  # config.json, pytorch_model.bin, tokenizer.json ...
└── bge-reranker-large/      # 可选，没有也能跑（缓存退化为词面匹配）
```

**重排模型是可选的**：不下载也不会报错，启动日志里会看到 `rerank.state = unavailable`，
问答缓存只走前两层。想榨干命中率再下。

### 安装后端

```powershell
cd backend

# 1. 建虚拟环境
py -m venv venv

# 2. 装 PyTorch（二选一，必须先装它，因为 CPU 版和 CUDA 版不是一个包）
#    CPU 版：约 120MB，能跑，编码慢一些
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#    CUDA 12.1 版：约 2.5GB
# .\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 装其余依赖
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

验证 PyTorch：

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 是 `False` 也能正常使用，只是慢。

### 配置文件

复制模板，改成你自己的值：

```powershell
# 在项目根目录
copy .env.example .env.local
```

至少要改这三项：

```ini
BGE_MODEL_PATH=H:/models/bge-m3        # 换成你的实际路径
DEEPSEEK_API_KEY=sk-xxxxxxxx           # 你的 Key
MYSQL_PORT=3307                        # 用 docker-compose.dev.yml 起库就是 3307
```

**配置优先级**（后者覆盖前者）：

```
代码里的默认值  <  .env / .env.local  <  config.override.json  <  系统环境变量
```

- `.env.local` 是最省事的方式，已被 `.gitignore` 排除
- `config.override.json` 适合放纯 JSON 的本机覆盖（同样被忽略），文件里的键名大小写不敏感
- 临时改一项可以直接设环境变量，例如 `$env:BGE_DEVICE="cpu"`，无需改文件

### 启动服务

**先自检**，它会检查依赖、模型路径、MySQL / Milvus 连通性、API Key 是否可用：

```powershell
cd backend
.\venv\Scripts\python.exe check_env.py
```

看到 `SUCCESS | 全部检查通过` 就说明环境没问题。然后开两个终端：

```powershell
# 终端 1：后端
cd backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
```

> `--reload-dir app` 很重要：不加的话 uvicorn 会监听整个 `backend/`，
> 你改一下 `tests/` 里的脚本就会触发后端热重启，正在跑的测试会撞上半路重启。

```powershell
# 终端 2：前端
cd frontend
npm install
npm run dev
```

打开 <http://localhost:3000>。API 文档在 <http://127.0.0.1:8000/api/v1/docs>。

> 首次启动会加载 BGE-M3（约 20~60 秒，取决于磁盘），日志里出现
> `嵌入模型预热完成` 之后再去传文档，体验会好很多。重排模型在后台慢慢加载，不影响使用。

---

## 配置项全表

都在 `backend/app/config.py` 的 `Settings` 里，用大写下划线同名环境变量覆盖。

### 基础

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENVIRONMENT` | `development` | `production` 时会关掉调试行为 |
| `DEBUG` | `true` | 影响日志级别与 reload |
| `API_PREFIX` | `/api/v1` | 接口前缀 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 后端监听地址 |
| `CORS_ORIGINS` | `["http://127.0.0.1:3000","http://localhost:3000"]` | 前端跨域白名单 |

### 模型

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BGE_MODEL_PATH` | `<root>/models/bge-m3` | 嵌入模型目录 |
| `BGE_DEVICE` | `cuda` | 改成 `cpu` 也能跑 |
| `BGE_BATCH_SIZE` | `16` | 显存吃紧就调小到 `8` / `4` |
| `BGE_MAX_LENGTH` | `1024` | 单块最大 token |
| `DEEPSEEK_API_KEY` | 空 | **必填** |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 换其他 OpenAI 兼容服务时改这里 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 也可用 `deepseek-reasoner` |
| `LLM_TEMPERATURE` | `0.3` | |
| `LLM_MAX_TOKENS` | `2048` | 注意：推理类模型会先消耗 token 在思维链上，给它留足够预算 |

### 存储

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | `127.0.0.1` / `3307` / `root` / `root` / `knowledge_manager` | 库不存在会自动建 |
| `MILVUS_HOST/PORT/COLLECTION` | `127.0.0.1` / `19530` / `km_chunks` | 集合与 HNSW 索引自动创建 |
| `DOCUMENTS_PATH` / `UPLOAD_PATH` | `<root>/data/documents`、`<root>/data/uploads` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 按字符切分 |

### 检索与文件名匹配

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SEARCH_TOP_K` | `5` | 向量召回条数 |
| `SCORE_THRESHOLD` | `0.0` | 低于该分数的分块直接丢弃 |
| `EMBED_WITH_NAME` | `true` | 把文件名/标题拼进待嵌入文本 |
| `FILENAME_MATCH_THRESHOLD` | `0.45` | 名称相似度达到即算"文件名命中"，参与加分 |
| `FILENAME_STRONG_THRESHOLD` | `0.6` | 强命中，直接保底入选 |
| `FILENAME_BOOST` | `0.20` | 命中时给向量分加的最大值 |
| `GRAPH_NAME_EDGE_THRESHOLD` | `0.6` | 图谱里"文件名关联"边的连边阈值 |

### 问答缓存

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QA_CACHE_ENABLED` | `true` | 总开关 |
| `QA_CACHE_SIMILARITY_THRESHOLD` | `0.85` | 词面（IDF 加权余弦）判定阈值 |
| `QA_CACHE_TOP_K` | `8` | 词面通道召回条数 |
| `QA_CACHE_TTL_DAYS` | `7` | 条目过期时间 |
| `QA_CACHE_MAX_ENTRIES` | `2000` | 缓存上限，超了淘汰最旧的 |
| `QA_CACHE_MIN_QUESTION_LEN` | `4` | 太短的问题不缓存（"好的" 之类） |
| `QA_CACHE_SEMANTIC_RECALL` | `0.70` | 语义通道**召回**阈值（不用于判定） |
| `QA_RERANKER_ENABLED` | `true` | 关掉就只用前两层 |
| `QA_RERANKER_PATH` | `<root>/models/bge-reranker-large` | 重排模型目录 |
| `QA_RERANK_THRESHOLD` | `0.995` | 语义最终判定阈值，见[设计说明](#设计说明问答缓存的三层命中) |
| `QA_RERANK_CANDIDATES` | `3` | 每轮最多重排几条（每条约 40ms） |

### 问答 agent

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QA_AGENT_ENABLED` | `true` | 总开关；`false` 退回固定单次检索（改造前行为） |
| `QA_AGENT_MAX_STEPS` | `3` | 证据收集最多几轮 LLM 往返（一轮内可并行多个工具调用） |
| `QA_AGENT_SCORE_GATE` | `0.55` | 单次检索最高分低于该值就升级成 agent 换措辞重搜 |
| `QA_AGENT_ALWAYS` | `true` | 忽略分数闸门，实质性问题一律走 agent 循环（用延迟换质量）；关键词/短答请求仍走单次检索 |
| `QA_QUERY_PLAN_ENABLED` | `true` | 作答前先规划检索策略（直接查 / 假设答案 / 拆子问题） |
| `QA_HYDE_MAX_TOKENS` | `220` | 「假设答案」检索词的生成预算 |
| `QA_TOOL_TRANSPORT` | `local` | 工具调用方式：`local` 直调（默认）/ `mcp` 走 MCP 协议（实验性，见[设计说明](#mcp)） |

### 历史对话

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HISTORY_MAX_CONVERSATIONS` | `200` | 会话数上限 |
| `HISTORY_MAX_MESSAGES` | `200` | 单会话消息数上限 |
| `HISTORY_CONTEXT_ROUNDS` | `3` | 追问时带几轮上下文，`0` 表示不带 |
| `HISTORY_CONTEXT_CHARS` | `400` | 每轮上下文截断长度 |

---

## 主要接口

完整交互式文档：<http://127.0.0.1:8000/api/v1/docs>

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档，后台异步索引 |
| GET | `/api/v1/documents` | 文档列表（分页 / 按状态筛选 / 按名称搜索） |
| GET | `/api/v1/documents/{id}/content` | 正文预览 |
| GET | `/api/v1/documents/{id}/summary-trace` | 摘要 Agent 的决策轨迹（读了哪几段、每步用了多少字） |
| PATCH | `/api/v1/documents/{id}` | 改文件名或标题（会即时重算该文档向量） |
| DELETE | `/api/v1/documents/{id}` | 删除（向量 + 库记录 + 磁盘文件） |
| POST | `/api/v1/documents/{id}/reindex` | 重新索引 |
| POST | `/api/v1/search` | 语义检索 |
| POST | `/api/v1/qa` | 问答（一次性返回 JSON） |
| POST | `/api/v1/qa/stream` | 问答（SSE 流式，事件：`meta` / `step` / `delta` / `error` / `done`） |
| GET | `/api/v1/graph` | 知识图谱（支持 `min_score` 阈值参数） |
| GET | `/api/v1/stats` | 知识库统计 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/health/detail` | 依赖健康检查（MySQL / Milvus / LLM / 嵌入模型） |
| GET | `/api/v1/agents` | Agent 列表与状态 |
| GET | `/api/v1/cache/stats` | 问答缓存统计（含重排模型状态） |
| GET | `/api/v1/cache` | 缓存条目列表 |
| DELETE | `/api/v1/cache` | 清空缓存 |
| GET | `/api/v1/conversations` | 会话列表 |
| GET | `/api/v1/conversations/{id}` | 会话详情（含消息） |
| PATCH | `/api/v1/conversations/{id}` | 重命名会话 |
| DELETE | `/api/v1/conversations/{id}` | 删除会话 |

流式问答示例：

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/qa/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"文档是怎么切分成块的","save_history":false}'
```

---

## 测试

项目没有用 pytest。每个脚本自己打印 `[OK] / [FAIL]`，靠退出码表示成败，
在 `backend/` 目录下执行即可（脚本自己会把 `backend/` 加进 `sys.path`）：

```powershell
cd backend
.\venv\Scripts\python.exe tests\verify_filename.py
```

| 脚本 | 前置条件 | 覆盖内容 |
|------|----------|----------|
| `smoke_test.py` | — | 端到端：造样例文档 → 索引 → 检索 → 问答 → 图谱 → 清理，直接调 Agent 不走 HTTP |
| `test_api.py` | 后端 | HTTP 接口全链路，共 17 项 |
| `verify_qa_cache.py` | 后端 | 缓存：命中 / 守卫 / 语序变体 / 数字变体 / 知识库变更后失效，共 36 项 |
| `verify_cache_rerank.py` | 后端 + 重排模型 | 三层命中、数字与否定词守卫、最小差异对必须拒绝，共 28 项 |
| `verify_graph_cache.py` | 后端 | 图谱向量缓存：冷启动 / 复用 / 增量 / 删除级联，共 15 项 |
| `verify_filename.py` | 后端 | 文件名匹配：分词、相似度对称性、检索加分、图谱连边，共 45 项 |
| `verify_fixes.py` | 后端 + 前端 + Playwright | 轮询刷新、画布尺寸、tooltip 溢出，共 11 项 |
| `verify_history_ui.py` | 后端 + 前端 + Playwright | 会话侧栏：新建 / 重命名 / 删除 / 刷新保持 / 缓存徽章，共 20 项 |
| `verify_filename_ui.py` | 后端 + 前端 + Playwright | 文件名在检索结果与图谱里的展示、改名弹窗，共 19 项 |
| `ui_test.py` | 后端 + 前端 + Playwright | 全流程走查并截图，共 7 项 |
| `verify_summary_agent.py` | 仅 LLM | 摘要 Agent：结论埋在末尾时是否真的读到尾部、步数上限、重复读取防护、reduce 兜底、短文档走单次，共 16 项 |
| `verify_qa_agent.py` | 后端 + LLM | QA Agent：置信度闸门（高分直出 / 触发词 / 短追问）、agent 关闭降级、强制循环、轮次预算、同步 run、边界，共 10 项 |
| `eval_qa_retrieval.py` | 后端 | 检索层评测：12 单轮 + 3 多轮，检查关键词是否落在召回分块里 |
| `eval_qa_agent.py` | 后端 + LLM | 答案层评测：同一套指标拨 `QA_AGENT_ENABLED` 开关做 A/B，报告 steps 与延迟 |

跑 UI 测试前需要设两个环境变量（否则 Playwright 会报 `EPERM ... mkdtemp`）：

```powershell
# 在 backend/ 目录执行，.. 即项目根目录
$root = (Resolve-Path "..").Path
$env:TMP="$root\.tmp"; $env:TEMP="$root\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH="$root\.playwright"

.\venv\Scripts\python.exe -m pip install playwright
.\venv\Scripts\python.exe -m playwright install chromium
```

离线评测工具（在 `backend/` 下）：

```powershell
# 59 个人工标注问题对，比较词面 / 语义 / 交叉编码器 / 大模型判定的召回与精确率
.\venv\Scripts\python.exe evaluate_cache_matchers.py
.\venv\Scripts\python.exe evaluate_llm_judge.py

# 索引耗时基准（冷启动 vs 模型已预热）
.\venv\Scripts\python.exe measure_index.py
```

详见 [`backend/tests/README.md`](./backend/tests/README.md)。

---

## 目录结构

```
knowledge-manager/
├── backend/
│   ├── app/
│   │   ├── agents/          # IndexAgent / SearchAgent / QAAgent / SummaryAgent / RelationAgent
│   │   ├── api/             # FastAPI 路由：documents / qa / graph / conversations / cache / system
│   │   ├── db/              # 引擎、会话、建表、幂等补列
│   │   ├── models/          # ORM：Document / Chunk / QaCache / Conversation / DocVector
│   │   ├── services/        # 业务层：解析、切块、嵌入、Milvus、检索、缓存、重排、LLM
│   │   └── utils/           # 中文分词、文件名匹配、文本切分
│   ├── tests/               # 10 个验证脚本 + 说明
│   ├── check_env.py         # 环境自检
│   ├── evaluate_*.py        # 缓存匹配算法离线评测
│   ├── measure_index.py     # 索引耗时基准
│   ├── migrate_filename.py  # 老数据修复工具
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── views/           # Home / Documents / Chat / Search / Graph
│   │   ├── api/index.js     # axios 封装
│   │   └── router/
│   ├── nginx.conf           # 生产反向代理（含 SSE 所需配置）
│   └── Dockerfile
├── data/                    # documents/ 与 uploads/，运行时生成，不入库
├── models/                  # 模型权重，需自行下载，不入库
├── docker-compose.dev.yml   # 只起 MySQL + Milvus
├── docker-compose.prod.yml  # 全栈容器化
├── .env.example             # 环境变量模板
└── README.md
```

---

## 设计说明：问答缓存的三层命中

这部分是整个项目里最花心思的地方，因为"缓存命中"和"答错"的代价完全不对称：
**一次错误命中 = 用户拿到一个自信但错误的答案，自己还察觉不到；一次未命中只是多等几秒。**
所以策略是：宁可少命中，也不能错。

### 为什么不是一个阈值

最初的方案是 BM25 召回 + softmax 归一化 + 相似度 ≥ 0.85。这里有个误解要澄清：
**softmax 不是归一化器**。它的分母是当前候选集合，同一对问题在不同候选集合里算出来的分数不一样，
阈值没有可复现的含义。真正稳定、有界的是 **IDF 加权余弦相似度**，这才是第二层的判定依据。

三层分别是：

| 层 | 判据 | 典型耗时 | 负责什么 |
|----|------|---------|---------|
| ① 精确指纹 | 归一化后的问题文本 hash 完全相等 | ~3 ms | 重复问同一个问题 |
| ② 词面 | jieba 分词 → BM25 召回 → IDF 加权余弦 ≥ 0.85 | ~5 ms | 只差几个虚词、语序变体 |
| ③ 语义 | 向量召回 ≥ 0.70 → 交叉编码器重排 ≥ 0.995 | 40~110 ms | 同义改写（"怎么切分" vs "咋切分"） |

第 ③ 层的两个阈值分工不同：**0.70 只负责"捞候选"，不负责判定**。
向量相似度分不开"前端配置"和"后端配置"这种最小差异，所以最终判定交给交叉编码器。

### 阈值是标定出来的，不是拍脑袋

用 59 对人工标注的问题对（28 正 / 31 负）扫了一遍：

| 方案 | 召回率 | 精确率 |
|------|--------|--------|
| ① 只用词面 ≥ 0.85 | 25.0% | 100% |
| ② **词面 或（语义 ≥ 0.70 且 重排 ≥ 0.995）** | **39.3%** | **100%** |
| ③ 参考：重排阈值放宽到 0.98 | 50.0% | 93.3% |
| ④ 参考：重排阈值放宽到 0.95 | 75.0% | 84.0% |

结论：放弃 93% 的精确率去换 10% 的召回率不值得，所以定在 0.995。
这个数字是在**作者的 59 对样本**上标出来的，换数据集应该重新标一次。

### 数字与否定词守卫

打分之前先过两道硬规则，不通过直接判为未命中：

- **数字必须一致**：「报销标准 500 元」和「报销标准 800 元」词面上极像，但答案完全不同
- **否定词按词判断**：用 jieba 分词后匹配，而不是子串匹配。
  否则「还有**别**的办法吗」「有**没**有」会被误判成否定疑问

### 失效策略

缓存的 key 里带了知识库签名 `文档数|最大更新时间`。任何一篇文档被新增、删除或重建，
签名就会变，缓存自动整体失效——不用手工清，也不用担心答的是旧内容。

### 大模型二次判定为什么没上

也试过让 DeepSeek 判断"这两个问题是不是同一个意思"：召回 64.3%、精确 100%，
但单次判定要 **1.7 秒**，比重新生成答案还慢，所以没接进热路径。
评测代码留在 `evaluate_llm_judge.py`，想接的话改 `qa_cache_service` 的第三层即可。

---

## 设计说明：摘要为什么是一个 agent

这是全项目唯一用了 ReAct 工具循环的地方。理由不是"多 Agent 听起来厉害"，而是这个任务真的具备两个前提：

1. **关键信息不在开头**。实测本机 6 篇文档里，有 3 篇在 5900~7600 字之间，
   而旧实现固定只读前 2500 字——超过一半的内容摘要根本没看到。
2. **读哪里需要判断**。文档类型很杂（简历、问卷、技术方案），
   没有固定规则能决定该读哪几段。

所以 `SummaryAgent` 拿到的是四个工具，而不是一段截断的文本：

| 工具 | 作用 |
|------|------|
| `get_outline` | 分块总数 + 每块开头 60 字，先建立结构认知 |
| `read_chunks(start, end)` | 读原文，单次最多 6 块 |
| `search_in_document(query)` | 复用 BM25 在本文档内定位 |
| `finish(summary, keywords)` | 提交结果，任务结束 |

一次真实的运行（17 块的技术方案文档）：

```
step1 get_outline                          chars=0
step2 read_chunks {"start":0,"end":2}      chars=1291
step2 read_chunks {"start":15,"end":16}    chars=1933   ← 先去读结尾的结论
step3 read_chunks {"start":13,"end":14}    chars=2916
step3 read_chunks {"start":11,"end":12}    chars=3774
step4 finish                               chars=3774
```

**这就是 agent 和 workflow 的区别**：读哪三段的顺序，不是我写的，是它自己决定的。

### 代价与控制

| 项 | 单次摘要 | ReAct |
|----|----------|-------|
| LLM 调用 | 1 次 | 3~6 次 |
| 一篇 17 块文档的耗时 | ~2s | ~19s |
| 可复现性 | 完全确定 | 路径每次可能不同 |

四道闸门保证它不会失控（缺一条就会卡在 `processing`）：

- `max_steps = 6`：循环步数上限
- `max_read_chunks = 6`：单次读取上限，防止一次灌满上下文
- `max_total_chars = 20000`：累计读取预算，超了强制收尾
- **reduce 兜底**：步数用完仍没结果，就用已读内容做一次普通摘要，保证一定有输出

轨迹落在 `documents.summary_trace`，通过 `GET /api/v1/documents/{id}/summary-trace` 查看。
agent 的行为不可复现，**必须能复盘"它到底读了哪几段"**，否则摘要错了都没法定位。

前端在「文档管理」页点摘要单元格即可打开抽屉：摘要全文、关键词、以及每一步的
`动作 + 累计读取字数` 时间线，例如

```
查看文档结构        共 17 个分块（下标 0~16）
读取第 1~3 块       累计读取 1291 字
读取第 14~17 块     累计读取 2918 字   ← 先去读结尾的结论
读取第 12~13 块     累计读取 3776 字
提交摘要            累计读取 3776 字
```

### 为什么不是全量 agent

简单事实题（"月结余多少"）单次检索 2 秒就能答对，走完整工具循环要 8~10 秒。
所以 QA 只在「单次检索结构上做不到」的时候才升级成 agent（见下节），其余保持确定性——
**能论证必要性的地方才用，其余保持确定性**。

---

## 设计说明：QA 为什么是 agent + 置信度闸门

`QAAgent` 的检索工具选择**不是**确定的：问题里出现文件名走词法通道、否则走向量，
这条规则只覆盖了一半场景。另一半——追问里的指代（"那第二点呢"）、跨文档对比、
通读整篇、一次搜不到换措辞重试——没有固定规则能预先写死，只能让模型自己判断。
所以 QA 从固定工作流改成了 **ReAct 工具循环**。

### 四个工具

| 工具 | 作用 |
|------|------|
| `search_kb(query, top_k)` | 语义检索，可换措辞多次调用 |
| `search_by_filename(name)` | 按文件名/标题定位文档，拿到文档 id 和分块数 |
| `read_document(id, start, end)` | 读某篇文档的原文区间，用于通读或补全半句话 |
| `list_documents` | 列出所有文档，用于"有哪些"或跨文档对比 |

循环**只负责收集证据**（按 `(文档, 片段)` 去重），收够了调 `finish`，最终作答仍走原来的
流式链——这样既保住 SSE，又不会二次生成。

一次真实的多轮追问（用户上一句问损失函数，这一句"那第二点呢"）：

```
search_kb("优化器 比较 Adam SGD 学习率")       ← 从"那第二点呢"自己改写出的完整问题
search_by_filename("作业2-李昊燃")             ← 从历史推断用户指的哪份文档
read_document(96, 5~6)                        ← 去读实验表格
finish
```

**证据收集做什么完全由模型决定**——搜几次、用什么词、读哪篇，代码里没有任何预设。

### 置信度闸门：不是所有问题都付 agent 的代价

全量走循环，简单题也要 10 秒，不能接受。所以先做一次单次检索，再判断要不要升级：

| 触发条件 | 为什么单次检索做不到 |
|----------|----------------------|
| 检索零命中 / 最高分 < `QA_AGENT_SCORE_GATE` | 需要换措辞重搜 |
| 有历史 + 短追问（≤12 字）或指代词开头（那/这/它/其中…） | 需要先把指代展开成完整问题 |
| 含「总结/全文/对比/有哪些/列出/分别有…」 | 单次检索结构上给不了 |

刻意**不含「比较」**——"比较的损失函数是哪一个"是在问某个值，不是要做对比，
算进来只会让简单问题白白多花 4 秒。

### 三层兜底

1. `QA_AGENT_ENABLED=false` → 完全退回改造前行为（单次检索 + 作答）
2. 循环抛异常 → 自动降级为单次检索，响应里标 `degraded=true`
3. 循环一步工具都没调成 → 用单次检索结果兜底

### 先规划检索，再作答

同一个问题，最优检索方式并不相同。作答前先让模型做一次小规划（`QA_QUERY_PLAN_ENABLED`）：

| 策略 | 什么时候用 | 怎么做 |
|------|-----------|--------|
| `direct` | 问句本身就是好的检索词、术语明确 | 原问句直接检索 |
| `hyde` | 术语含糊、开放式概念题（"agent 是什么"） | 先生成一段假设答案，用它当检索词——答案的文体和文档更像 |
| `decompose` | 多个对象/多处来源要拼（"对比 A 和 B 的评测指标"） | 拆成 2-4 个子问题各自检索后合并 |

规划同时判断**问题类型**（`scope`）：

- `kb`：问的是知识库里的具体东西（某文档、某项目、某指标）
- `general`：问的是通用概念、知识库只有个别实现片段

`general` 时会往 prompt 里加一条提示：先给学科通识层，再落到知识库的实现层，
分别标注，不要拿某一份文档当全部答案。这条是规则触发的（命中
"是什么/什么是/解释/全面/深入/详细介绍…" 且不含"这份/这个项目"），
不靠模型自觉——实测模型自己判断时会把通用问题当成 `kb`。

规划失败（模型超时、JSON 不可解析、HyDE 空产出）一律回到 `direct`，不影响主链路。

实测（同一批问题）：

| 问题 | 规划 | 耗时 | 字/词数 | 结果 |
|------|------|------|--------|------|
| 帮我给《聚类.md》生成关键词 | direct（关键词短路） | 3.4s | 25 字 | 直接命中 |
| 对比简历审查与试卷批改的评测指标 | decompose（3 子问题） | 27.4s | 441 字 | 两方指标分开说清 |
| 详细讲解一下 agent | hyde + general | 39.5s | 1516 字 | 通识层（什么是/怎么搭/质量效率/工程化/评测边界）→ 落到知识库实现 |

代价：规划 + HyDE + 循环 + 提纲各多一次 LLM 往返，实质性问题从约 15 秒涨到 25-40 秒。
只要质量不想要延迟，把 `QA_AGENT_ALWAYS` 和 `QA_QUERY_PLAN_ENABLED` 关掉即回到旧行为。

### 实测（同一套 15 条评测，拨开关做 A/B）

| | agent 关 | agent 开 |
|---|---|---|
| 单轮 | 92%（2.8s） | 92%（4.2s） |
| **多轮追问** | **67%**（4.4s） | **100%**（8.0s） |
| 合计 | 87%（3.6s） | **93%**（6.1s） |

分数提升全在多轮追问。延迟是真代价：agent 问题的 6~9 秒比单次检索的 2 秒更让人等得难受，
所以前端把 `step` 事件渲染成「🔍 检索了什么 / 📄 读了哪几段」时间线，把等待变成可看的进度。

### MCP

`backend/app/mcp_server.py` 把同一套 `TOOL_SCHEMAS` 包成 stdio MCP server，可被
Claude Desktop 等宿主直接调用：

```json
{
  "knowledge-manager": {
    "command": "python",
    "args": ["-m", "app.mcp_server"],
    "env": { "PYTHONPATH": "F:/agent/agentuse/backend" }
  }
}
```

`QA_TOOL_TRANSPORT=mcp` 时，agent 内部改走 MCP 协议调工具（默认 `local` 直调，不付子进程开销）。

> ⚠️ **实验性**：`QA_TOOL_TRANSPORT=mcp`（内部 agent 走 MCP）目前是实验特性。
> 在 Windows 上，`search_kb` 这类需要加载 BGE 模型的工具会在 MCP 的 stdio 子进程里
> 卡死（`list_documents` / `read_document` 等纯 MySQL 工具正常），根因是 anyio 子进程
> 与 transformers 初始化的交互，尚未定位。默认的 `local` 直调不受影响，是生产路径。
> **外部宿主**（如 Claude Desktop）直接连 `mcp_server.py` 不受此问题影响。

---

## 常见问题

**端口被占用：`WinError 10013` / `address already in use`**

```powershell
# 看谁占了 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Get-Process -Id <PID>
# 或者用仓库里的脚本按端口清理
.\scripts\free-ports.ps1
```

**`torch.cuda.is_available()` 返回 False，但我有显卡**

1. 确认装的是 CUDA 版：`python -c "import torch; print(torch.version.cuda)"` 不该是 `None`
2. `nvidia-smi` 能正常输出
3. 不要混装 CPU 版和 CUDA 版：先 `pip uninstall torch` 再重装

**显存不够：`CUDA out of memory`**

BGE-M3 常驻约 2.2GB，重排模型再加约 2.2GB。8GB 显存要同时跑后端和另一个测试进程时容易 OOM。三个办法：

- 调小 `BGE_BATCH_SIZE`（16 → 8 → 4）
- 用环境变量临时切 CPU：`$env:BGE_DEVICE="cpu"`
- 关掉重排：`$env:QA_RERANKER_ENABLED="false"`

**Milvus 连不上 / `Fail connecting to server`**

```bash
docker compose -f docker-compose.dev.yml ps      # 看 milvus-standalone 是否 healthy
docker logs km_milvus_dev --tail 100
```

Milvus 自己还依赖 etcd 和 MinIO，起来大约要 30~60 秒，等 healthy 再启动后端。

**MySQL 端口到底是 3306 还是 3307**

- 用 `docker-compose.dev.yml` 起库 → 宿主机是 **3307**
- 自己本机装 MySQL → 默认 **3306**

对不上就会报 `Can't connect to MySQL server`。

**上传后一直显示"索引中"**

先看后端日志。常见原因是嵌入模型没加载完（首次要几十秒），或者 `BGE_MODEL_PATH` 指错了目录。
接口响应带 `Cache-Control: no-store`，所以不是浏览器缓存的问题。

**同一个文件传两次会怎样**

按内容 hash 去重：第二次上传会直接复用已有记录，不会重复入库，也不会在磁盘上留下多余副本。

**中文在 PowerShell 里显示成乱码**

那是 PowerShell 用 GBK 解码 UTF-8 文件的**显示**问题，文件本身没坏。
要看真实内容用：

```powershell
[System.IO.File]::ReadAllText("README.md", [System.Text.Encoding]::UTF8)
```

**没有显卡能跑吗**

能。`BGE_DEVICE=cpu`，只是编码慢 5~10 倍（一篇文章从 0.2 秒变成 1~2 秒），
问答本身主要还是等大模型，日常用不至于难受。

**重排模型缺失会怎样**

启动日志里 `rerank.state = unavailable`，`/api/v1/cache/stats` 里也能看到。
此时缓存只走前两层（仍能命中措辞几乎相同的问题），不会报错、不会阻塞请求。

---

## 安全提示与已知限制

这是一个**单用户、本机使用**的项目，请先看清它没有做什么：

- **没有认证授权**。任何能访问 8000 端口的人都可以读、删你的全部文档和缓存。
  默认只监听 `127.0.0.1`；**不要直接暴露到公网**。要对外提供访问，
  请在前面加一层带鉴权的反向代理（Nginx Basic Auth / OAuth2 Proxy 之类），并启用 HTTPS。
- **没有多租户**。所有人共享同一个知识库。
- **CORS 白名单**只放行了本地调试端口，换端口记得同步改 `CORS_ORIGINS`。
- **会话列表只支持 `limit`**，没有分页游标，超过 500 条会截断。
- **单进程**。后台索引用 FastAPI `BackgroundTasks`，重活会在请求之外跑，但没有任务队列，
  服务重启时正在索引的任务会丢（文档停留在"处理中"状态，需要手动重建索引）。
- **缓存判定是基于统计的**，不是形式化保证。极端情况下仍可能误命中，
  所以特意保留了数字 / 否定词守卫和保守阈值。

---

## 服务器部署（可选）

> 只是想在本地用的话，这一节可以跳过。

`docker-compose.prod.yml` 会把后端、前端、MySQL、Milvus 全容器化，
前端由 Nginx 提供并反代 `/api`（SSE 需要的 `proxy_buffering off` 已经配好）。

**1. 准备文件**

```bash
# 在服务器上
git clone <你的仓库地址> knowledge-manager && cd knowledge-manager
mkdir -p models data/documents data/uploads
# 把 bge-m3（和可选的 bge-reranker-large）放到 models/ 下
```

**2. 建 `.env.prod`**

`docker-compose.prod.yml` 会读取根目录的 `.env.prod`，这个文件**必须自己创建**（已在 `.gitignore` 里）。
参考下表填写：

| 变量 | 示例值 | 说明 |
|------|--------|------|
| `ENVIRONMENT` | `production` | |
| `DEBUG` | `false` | |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 容器内监听 |
| `MYSQL_HOST` | `mysql` | 用 compose 里的服务名 |
| `MYSQL_PORT` | `3306` | 容器内端口，不是宿主的 3307 |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `root` / 强口令 | |
| `MYSQL_ROOT_PASSWORD` | 同上 | **必填且无默认值**，没设会直接启动失败 |
| `MYSQL_DATABASE` | `knowledge_manager` | |
| `MILVUS_HOST` | `milvus-standalone` | |
| `BGE_MODEL_PATH` | `/app/models/bge-m3` | 对应宿主 `./models/bge-m3` |
| `BGE_DEVICE` | `cpu` | 要 GPU 就装 nvidia-container-toolkit 并放开 compose 里的注释 |
| `QA_RERANKER_PATH` | `/app/models/bge-reranker-large` | 没有就设 `QA_RERANKER_ENABLED=false` |
| `DEEPSEEK_API_KEY` | `sk-...` | |

**3. 启动**

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend
```

访问 `http://<服务器IP>/`，API 文档在 `http://<服务器IP>/api/v1/docs`。

> 仍然要提醒一次：镜像里**没有鉴权**。放到公网之前请套一层带认证的反向代理。

---

## 开发约定

- **改完代码先跑相关测试**。改动落在缓存上跑 `verify_qa_cache.py` / `verify_cache_rerank.py`，
  改文件名逻辑跑 `verify_filename.py`，改前端跑对应的 UI 脚本。
- **测试脚本必须可重复运行**：自己造的数据自己清。
- **接口响应不加缓存**（已有中间件统一处理），否则前端轮询会看到旧状态。
- **中文注释**，但 `requirements.txt` 保持纯 ASCII —— Windows 上 pip 会用系统本地编码读它。
- 不要用 `Get-Content ... | Set-Content ...` 回写中文文件，
  PowerShell 5.1 会用 GBK 读写，把 UTF-8 文件写坏。

---

## License

[MIT](./LICENSE)
