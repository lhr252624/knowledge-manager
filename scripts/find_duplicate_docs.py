"""
近重复文档检测（维护脚本，不参与线上请求）

为什么要它：
    同一个知识库里往往躺着同一批笔记的多个版本（"0728 重点复习""0806 内容总结"
    "第四章 复习笔记"…）。检索时它们会各贡献两三个片段，模型拿到的是同一件事的
    不同说法拼起来的碎片，读起来就是"每段都对、放一起不知道在说什么"。

做法：
    把每篇文档的分块正文拼起来，按字符 3-gram 做集合，两两算 Jaccard / 包含度。
    纯 CPU、不占显存，所以可以在后端运行时跑。

用法：
    python scripts/find_duplicate_docs.py                 # 只看分布，不写文件
    python scripts/find_duplicate_docs.py --threshold 0.5 --write
"""
import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.database import session_scope  # noqa: E402
from app.models.document import Chunk, Document, DocumentStatus  # noqa: E402

KEEP = set("0123456789abcdefghijklmnopqrstuvwxyz")
MAX_CHARS = 20000


def normalize(text: str) -> str:
    """只保留中英文数字，其余（空格、标点、换行）全部丢掉"""
    out = []
    for ch in (text or "").lower():
        if ch in KEEP or "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    return "".join(out)


def shingles(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def load_documents():
    with session_scope() as db:
        docs = (
            db.query(Document.id, Document.title, Document.filename, Document.chunk_count)
            .filter(Document.status == DocumentStatus.READY)
            .all()
        )
        items = []
        for doc_id, title, filename, chunk_count in docs:
            rows = (
                db.query(Chunk.content)
                .filter(Chunk.document_id == doc_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            body = "\n".join(r.content or "" for r in rows)[:MAX_CHARS]
            if len(normalize(body)) < 200:
                continue
            items.append(
                {
                    "id": int(doc_id),
                    "title": (title or filename or f"doc{doc_id}")[:60],
                    "chunks": int(chunk_count or 0),
                    "shingles": shingles(normalize(body)),
                }
            )
    return items


def score(a: set, b: set):
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    if not inter:
        return 0.0, 0.0
    jaccard = inter / len(a | b)
    containment = inter / min(len(a), len(b))  # 一篇是另一篇的子集时也能抓到
    return jaccard, containment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "backend" / "data" / "doc_duplicates.json"))
    args = parser.parse_args()

    docs = load_documents()
    print(f"参与比较的文档：{len(docs)} 篇\n")

    pairs = []
    for a, b in combinations(docs, 2):
        jac, con = score(a["shingles"], b["shingles"])
        if jac <= 0:
            continue
        pairs.append((jac, con, a, b))
    pairs.sort(key=lambda x: -x[1])

    print("按 Jaccard 排前 25 对：")
    print(f"{'jaccard':>8} {'contain':>8}  文档A / 文档B")
    for jac, con, a, b in pairs[:25]:
        print(f"{jac:8.3f} {con:8.3f}  [{a['id']}] {a['title']}  <->  [{b['id']}] {b['title']}")

    # 并查集聚类：Jaccard >= threshold 视为同一簇
    parent = {d["id"]: d["id"] for d in docs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked = []
    for jac, con, a, b in pairs:
        if jac >= args.threshold:
            ra, rb = find(a["id"]), find(b["id"])
            if ra != rb:
                parent[rb] = ra
            linked.append((jac, a["id"], b["id"]))

    groups = {}
    for d in docs:
        groups.setdefault(find(d["id"]), []).append(d)

    clusters = [g for g in groups.values() if len(g) > 1]
    print(f"\n阈值 {args.threshold} 下：{len(clusters)} 个近重复簇，共 {sum(len(g) for g in clusters)} 篇")
    duplicates = {}
    for g in sorted(clusters, key=lambda g: -len(g)):
        # 代表 = 分块最多的那篇（内容最全），同分取 id 小的
        rep = sorted(g, key=lambda d: (-d["chunks"], d["id"]))[0]
        print(f"\n  代表：[{rep['id']}] {rep['title']}（{rep['chunks']} 块）")
        for d in g:
            if d["id"] == rep["id"]:
                continue
            duplicates[str(d["id"])] = rep["id"]
            print(f"    - [{d['id']}] {d['title']}（{d['chunks']} 块）")

    if args.write:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": args.threshold,
            "doc_count": len(docs),
            "duplicates": duplicates,
            "clusters": [
                {
                    "rep": sorted(g, key=lambda d: (-d["chunks"], d["id"]))[0]["id"],
                    "members": [d["id"] for d in g],
                }
                for g in clusters
            ],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}（{len(duplicates)} 篇被标记为重复）")


if __name__ == "__main__":
    main()
