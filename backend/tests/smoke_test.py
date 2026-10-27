"""
端到端冒烟测试

不启动 HTTP 服务，直接调用 Agent 验证完整链路：
    造样例文档 -> IndexAgent 索引 -> SearchAgent 检索 -> QAAgent 问答
    -> RelationAgent 图谱 -> 清理

用法：
    cd backend
    .\\venv\\Scripts\\python.exe smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <7}</level> | {message}")

SAMPLE_1 = """# 智慧校园知识库建设方案

## 一、背景

学校现有各类文档分散存放于教师个人电脑与网盘中，检索困难、版本混乱。
本方案提出建设统一的智慧校园知识库平台，实现文档集中管理与智能检索。

## 二、技术架构

平台采用前后端分离架构：
- 后端使用 Python FastAPI 框架，负责文档解析、向量化与检索接口。
- 向量数据库选用 Milvus，采用 HNSW 索引与内积度量。
- 关系数据库使用 MySQL 8.0，存储文档元数据与分块文本。
- 嵌入模型采用 BGE-M3，输出 1024 维向量。
- 大模型接入 DeepSeek，提供问答与摘要能力。

## 三、核心功能

1. 文档自动索引：支持 PDF、Word、Markdown、纯文本。
2. 语义检索：基于向量相似度召回相关片段。
3. 智能问答：检索增强生成，回答附带来源引用。
4. 知识图谱：按文档语义相似度自动连边。

## 四、实施计划

第一阶段完成环境搭建与文档索引；第二阶段上线检索与问答；
第三阶段补充知识图谱与权限管理。预计三个月完成验收。
"""

SAMPLE_2 = """# 项目部署与运维手册

## 部署方式

本项目使用 Docker Compose 进行编排。开发环境仅启动数据库服务，
后端与前端在宿主机直接运行以便热重载。生产环境采用全栈容器化部署。

## 服务端口

MySQL 默认映射到宿主机 3307 端口，用于避开本机已有的 MySQL 实例。
Milvus 使用 19530 端口，前后端分别暴露 80 与 8000。

## 备份策略

每周对 MySQL 数据目录做一次全量备份，Milvus 数据卷随镜像定期快照。
上传的原始文档保存在 data/uploads 目录，建议纳入版本外的增量备份。

## 故障处理

若 Milvus 容器反复重启，通常是 etcd 或 MinIO 尚未健康，需等待其就绪后重启。
若文档索引长期停留在处理中状态，检查后端日志中的解析异常。
"""


def make_sample_files(tmpdir: Path) -> list[Path]:
    files = []
    for i, content in enumerate((SAMPLE_1, SAMPLE_2), start=1):
        p = tmpdir / f"sample_{i}.md"
        p.write_text(content, encoding="utf-8")
        files.append(p)
    return files


def main() -> int:
    from app.agents.index_agent import index_agent
    from app.agents.qa_agent import qa_agent
    from app.agents.relation_agent import relation_agent
    from app.agents.search_agent import search_agent
    from app.config import get_settings
    from app.db.database import init_db
    from app.services.document_service import delete_document, get_stats

    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    created_ids: list[int] = []

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        files = make_sample_files(tmpdir)

        # ---------- 1. 索引 ----------
        print("\n" + "=" * 60)
        print("  [1/5] IndexAgent 索引文档")
        print("=" * 60)
        for f in files:
            result = index_agent.run(file_path=str(f), generate_summary=True)
            created_ids.append(result["document_id"])
            print(
                f"  文档 {result['document_id']}: {result.get('filename')} "
                f"| 分块 {result.get('chunk_count')} "
                f"| 摘要 {len(result.get('summary') or '')} 字 "
                f"| 关键词 {result.get('keywords')}"
            )

        # ---------- 2. 检索 ----------
        print("\n" + "=" * 60)
        print("  [2/5] SearchAgent 语义检索")
        print("=" * 60)
        res = search_agent.run(query="向量数据库用的是什么？", top_k=5)
        print(f"  命中片段 {res['total_chunks']} 条")
        for doc in res["documents"]:
            print(
                f"  - {doc['title']} | 最高分 {doc['best_score']:.3f} "
                f"| 命中 {doc['hit_count']} 处"
            )

        # ---------- 3. 问答 ----------
        print("\n" + "=" * 60)
        print("  [3/5] QAAgent 问答")
        print("=" * 60)
        qa = qa_agent.run(question="这个项目用了哪些数据库和模型？", top_k=5)
        print("  答案：")
        for line in (qa["answer"] or "").split("\n"):
            print(f"    {line}")
        print(f"  引用 {len(qa['citations'])} 条")

        # ---------- 4. 图谱 ----------
        print("\n" + "=" * 60)
        print("  [4/5] RelationAgent 知识图谱")
        print("=" * 60)
        graph = relation_agent.run(min_score=0.5)
        print(f"  节点 {len(graph['nodes'])} / 边 {len(graph['edges'])}")
        for e in graph["edges"]:
            print(f"  - {e['source']} <-> {e['target']} 相似度 {e['value']}")

        # ---------- 5. 统计 ----------
        print("\n" + "=" * 60)
        print("  [5/5] 统计与清理")
        print("=" * 60)
        stats = get_stats()
        print(f"  文档 {stats['total_documents']} | 分块 {stats['total_chunks']} "
              f"| 向量 {stats['total_vectors']}")

        for doc_id in created_ids:
            delete_document(doc_id)
        print(f"  已清理测试文档 {created_ids}")

    print("\n" + "=" * 60)
    logger.success("冒烟测试全部通过")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception(f"冒烟测试失败: {exc}")
        sys.exit(1)
