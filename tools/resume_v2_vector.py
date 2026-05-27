#!/usr/bin/env python3
"""Resume ResearchKB v2 Chroma vector ingestion from chunks.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]


def load_existing_ids(vector_db: Path) -> set[str]:
    sqlite_path = vector_db / "chroma.sqlite3"
    if not sqlite_path.exists():
        return set()
    con = sqlite3.connect(sqlite_path)
    try:
        return {row[0] for row in con.execute("SELECT embedding_id FROM embeddings")}
    finally:
        con.close()


def iter_missing_chunks(chunks_path: Path, existing_ids: set[str]):
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_id"] not in existing_ids:
                yield chunk


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def count_texts(path: Path) -> int:
    return len(list(path.glob("*.txt"))) if path.exists() else 0


def write_manifest(out_root: Path, elapsed_seconds: float, collection_count: int, total_chunks: int) -> None:
    papers = json.loads((out_root / "metadata" / "all_papers.json").read_text(encoding="utf-8"))
    manifest = {
        "version": "v2",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "papers": len(papers),
        "aaai_papers": sum(1 for p in papers if str(p.get("venue", "")).startswith("AAAI")),
        "papers_with_text": count_texts(out_root / "texts"),
        "papers_with_abstract": sum(1 for p in papers if p.get("abstract")),
        "chunks": total_chunks,
        "vector_count": collection_count,
        "fts_index": str(out_root / "fts_index.sqlite"),
        "vector_db": str(out_root / "vector_db"),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume v2 vector ingestion")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_KB_ROOT / "v2")
    parser.add_argument("--model-name", default=os.environ.get("RESEARCH_KB_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--collection", default="research_papers_v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    out_root = args.out_root.resolve()
    chunks_path = out_root / "chunks.jsonl"
    vector_db = out_root / "vector_db"
    total_chunks = count_lines(chunks_path)
    existing_ids = load_existing_ids(vector_db)
    print(f"total_chunks={total_chunks} existing={len(existing_ids)} missing={total_chunks - len(existing_ids)}", flush=True)

    import chromadb
    import torch
    from chromadb.utils import embedding_functions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model_name, device=device)
    client = chromadb.PersistentClient(path=str(vector_db))
    collection = client.get_or_create_collection(name=args.collection, embedding_function=emb_fn)

    ids, documents, metadatas = [], [], []
    done = len(existing_ids)
    for chunk in iter_missing_chunks(chunks_path, existing_ids):
        section = chunk.get("section", "")
        prefix = f"[{chunk.get('venue', '')} {chunk.get('year', '')}] {chunk.get('title', '')}\nSection: {section}\n"
        ids.append(chunk["chunk_id"])
        documents.append(prefix + chunk["text"])
        metadatas.append(
            {
                "paper_id": chunk["paper_id"],
                "title": chunk.get("title", ""),
                "venue": chunk.get("venue", ""),
                "year": chunk.get("year", ""),
                "section": section,
                "chunk_type": chunk.get("chunk_type", ""),
                "chunk_index": int(chunk.get("chunk_index") or 0),
            }
        )
        if len(ids) >= args.batch_size:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            done += len(ids)
            print(f"vector {done}/{total_chunks}", flush=True)
            ids, documents, metadatas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        done += len(ids)
        print(f"vector {done}/{total_chunks}", flush=True)

    count = collection.count()
    print(f"vector_count={count} device={device}", flush=True)
    if count == total_chunks:
        write_manifest(out_root, time.time() - started, count, total_chunks)
    else:
        raise SystemExit(f"incomplete vector index: {count}/{total_chunks}")


if __name__ == "__main__":
    main()
