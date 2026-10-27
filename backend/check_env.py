"""
环境自检脚本

用法：
    cd backend
    .\\venv\\Scripts\\python.exe check_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <7}</level> | {message}")


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(f"  {title}")
    print("=" * 58)


def main() -> int:
    results: dict[str, bool] = {}

    # ---------- 1. 配置 ----------
    section("1/6 配置加载")
    try:
        from app.config import get_settings

        s = get_settings()
        print(f"  环境      : {s.environment}")
        print(f"  BGE 模型  : {s.bge_model_path}")
        print(f"  BGE 设备  : {s.bge_device}")
        print(f"  LLM       : {s.deepseek_model}")
        print(f"  API Key   : {'已配置' if s.deepseek_api_key else '缺失'}")
        print(f"  MySQL     : {s.mysql_host}:{s.mysql_port}/{s.mysql_database}")
        print(f"  Milvus    : {s.milvus_host}:{s.milvus_port}")
        results["配置"] = bool(s.deepseek_api_key) and Path(s.bge_model_path).is_dir()
        if not results["配置"]:
            print("  [问题] 检查 API Key 与 BGE 模型路径")
    except Exception as exc:
        logger.error(f"配置加载失败: {exc}")
        results["配置"] = False

    # ---------- 2. MySQL ----------
    section("2/6 MySQL 连接")
    try:
        from app.db.database import check_mysql, init_db

        ok = check_mysql()
        print(f"  连接: {'成功' if ok else '失败'}")
        if ok:
            init_db()
            print("  数据表: 已就绪")
        results["MySQL"] = ok
    except Exception as exc:
        logger.error(f"MySQL 检查失败: {exc}")
        results["MySQL"] = False

    # ---------- 3. Milvus ----------
    section("3/6 Milvus 向量库")
    try:
        from app.services.milvus_store import milvus_store

        ok = milvus_store.health()
        print(f"  连接: {'成功' if ok else '失败'}")
        if ok:
            print(f"  集合: {milvus_store.collection.name}")
            print(f"  向量数: {milvus_store.count()}")
        results["Milvus"] = ok
    except Exception as exc:
        logger.error(f"Milvus 检查失败: {exc}")
        results["Milvus"] = False

    # ---------- 4. BGE 模型 ----------
    section("4/6 BGE-M3 嵌入模型")
    try:
        from app.services.embedding_service import embedding_service

        vec = embedding_service.encode("这是一段用于测试的中文文本")
        dim = len(vec)
        print(f"  设备  : {embedding_service.device}")
        print(f"  维度  : {dim}")
        print(f"  示例  : {[round(v, 4) for v in vec[:5]]} ...")
        results["BGE"] = dim == 1024
    except Exception as exc:
        logger.error(f"BGE 模型加载失败: {exc}")
        results["BGE"] = False

    # ---------- 5. DeepSeek ----------
    section("5/6 DeepSeek LLM")
    try:
        from app.services.llm_service import llm_service

        ok = llm_service.health()
        results["LLM"] = ok
        print(f"  调用: {'成功' if ok else '失败'}")
    except Exception as exc:
        logger.error(f"LLM 调用失败: {exc}")
        results["LLM"] = False

    # ---------- 6. 分块器 ----------
    section("6/6 文本分块器")
    try:
        from app.utils.text_splitter import split_text

        sample = "第一段内容。" * 60 + "\n\n" + "第二段内容。" * 60
        pieces = split_text(sample, chunk_size=200, chunk_overlap=30)
        print(f"  输入长度: {len(sample)}")
        print(f"  切块数量: {len(pieces)}")
        print(f"  首块长度: {len(pieces[0]['content'])}")
        results["分块器"] = len(pieces) > 1
    except Exception as exc:
        logger.error(f"分块器异常: {exc}")
        results["分块器"] = False

    # ---------- 汇总 ----------
    section("检查结果")
    all_ok = True
    for name, ok in results.items():
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        logger.success("全部检查通过，可以启动服务")
    else:
        logger.warning("存在未通过项，请先修复再启动服务")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
