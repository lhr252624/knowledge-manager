"""
文件名迁移脚本

解决两个历史问题：

    1. 早期版本上传时把磁盘上的 UUID 名当成了文件名入库，
       documents.filename 里是 "a84b6fef40264cdfa2db8cf839d6dece.md" 这种东西。
       原始名字在旧日志里没有记录，无法找回，这里用「标题 + 原扩展名」重建。

    2. 早期版本的分块向量只编码了正文，文件名/标题完全不参与。
       现在 insert_chunks 会把名字拼进嵌入文本，历史向量需要重算。

脚本是幂等的：已经修好的文档会被跳过；加 --force 可强制重算。

用法:
    venv\\Scripts\\python.exe migrate_filename.py --dry-run   # 只看会改什么
    venv\\Scripts\\python.exe migrate_filename.py             # 执行
    venv\\Scripts\\python.exe migrate_filename.py --force      # 全部重算向量
"""
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))


from app.db.database import session_scope  # noqa: E402
from app.models.document import Chunk, Document, DocumentStatus  # noqa: E402
from app.services.milvus_store import milvus_store  # noqa: E402
from app.utils.name_matcher import is_placeholder_name  # noqa: E402


def _derive_filename(title: str | None, file_type: str | None, stored_name: str) -> str:
    """用标题 + 原扩展名重建文件名"""
    ext = (Path(stored_name).suffix or f".{(file_type or 'txt').lstrip('.')}").lower()
    clean = (title or "").strip().replace("/", "-").replace("\\", "-")
    if not clean:
        return f"未命名文档{ext}"
    return f"{clean[:120]}{ext}"


def step1_backfill_names(dry_run: bool) -> List[int]:
    """回填 documents.filename"""
    fixed: List[int] = []
    with session_scope() as db:
        docs = db.query(Document).order_by(Document.id).all()
        for doc in docs:
            if not is_placeholder_name(doc.filename):
                continue
            new_name = _derive_filename(doc.title, doc.file_type, doc.stored_name)
            print(f"  doc {doc.id:<3} {doc.filename}  ->  {new_name}")
            fixed.append(doc.id)
            if not dry_run:
                doc.filename = new_name
    return fixed


def step2_reembed(document_ids: List[int], dry_run: bool) -> int:
    """重算分块向量（把文件名/标题并入嵌入文本）"""
    from app.services.document_service import reembed_document

    total = 0
    for doc_id in document_ids:
        with session_scope() as db:
            doc = db.get(Document, doc_id)
            if doc is None:
                continue
            filename = doc.filename
            count = db.query(Chunk).filter(Chunk.document_id == doc_id).count()

        if not count:
            print(f"  doc {doc_id} 没有分块，跳过")
            continue

        print(f"  doc {doc_id} 重新编码 {count} 个分块（{filename}）")
        if dry_run:
            total += count
            continue

        total += reembed_document(doc_id)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="文件名回填 + 分块向量重算")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不修改")
    parser.add_argument(
        "--force", action="store_true", help="所有文档都重算向量（默认只处理名字被修改过的）"
    )
    parser.add_argument(
        "--only", type=str, default="", help="只处理指定 id，逗号分隔，如 5,6"
    )
    args = parser.parse_args()

    only = [int(x) for x in args.only.split(",") if x.strip().isdigit()]

    with session_scope() as db:
        all_ids = [
            d.id
            for d in db.query(Document)
            .filter(Document.status == DocumentStatus.READY)
            .order_by(Document.id)
            .all()
        ]
    if only:
        all_ids = [i for i in all_ids if i in only]

    print("=" * 70)
    print("步骤 1/2  回填被 UUID 覆盖的文件名")
    print("=" * 70)
    fixed = step1_backfill_names(args.dry_run)
    if not fixed:
        print("  无需回填")

    print()
    print("=" * 70)
    print("步骤 2/2  重算分块向量（文件名 + 标题并入嵌入文本）")
    print("=" * 70)
    targets = sorted(set(fixed) | (set(all_ids) if args.force else set()))
    if not targets:
        print("  无需重算（加 --force 可强制全部重算）")
    else:
        total = step2_reembed(targets, args.dry_run)
        print(f"  共处理 {len(targets)} 篇文档 / {total} 个分块")

    print()
    print("=" * 70)
    if args.dry_run:
        print("  [dry-run] 以上均为预览，未写入任何数据")
    else:
        print(f"  Milvus 当前向量数: {milvus_store.count()}")
        with session_scope() as db:
            docs = db.query(Document).order_by(Document.id).all()
            print("  当前文档列表:")
            for d in docs:
                flag = "  <-- 仍是占位名" if is_placeholder_name(d.filename) else ""
                print(f"    {d.id:<3} {d.status:<8} {d.filename}{flag}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
