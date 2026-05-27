#!/usr/bin/env python3
"""
Rebuild ResearchKB v2 indexes.

Outputs under research_kb/v2:
  - metadata/all_papers.json
  - metadata/download_map.json
  - texts/<paper_id>.txt
  - chunks.jsonl
  - fts_index.sqlite
  - vector_db/ Chroma collection
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]
SECTION_PATTERNS = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.I)),
    ("introduction", re.compile(r"^\s*(?:\d+\.?\s*)?introduction\s*$", re.I)),
    ("related_work", re.compile(r"^\s*(?:\d+\.?\s*)?(?:related work|background)\s*$", re.I)),
    ("method", re.compile(r"^\s*(?:\d+\.?\s*)?(?:method|methods|methodology|approach|proposed method|model|framework)\s*$", re.I)),
    ("experiments", re.compile(r"^\s*(?:\d+\.?\s*)?(?:experiments?|experimental results|results|evaluation)\s*$", re.I)),
    ("conclusion", re.compile(r"^\s*(?:\d+\.?\s*)?(?:conclusion|conclusions|discussion)\s*$", re.I)),
    ("references", re.compile(r"^\s*(?:references|bibliography)\s*$", re.I)),
]
SKIP_INDEX_SECTIONS = {"references"}


@dataclass
class ExtractedText:
    text: str
    pages: int


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def clean_text(text: str) -> str:
    text = re.sub(r"<latexit[^>]*>.*?</latexit>", "", text, flags=re.DOTALL)
    text = re.sub(r"[A-Za-z0-9+/=]{80,}", "", text)
    text = re.sub(r"[^\x00-\x7F\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]{8,}", "", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def safe_paper_id(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("_.") or hashlib.md5(paper_id.encode()).hexdigest()


def resolve_path(repo_root: Path, kb_root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = (
        [path]
        if path.is_absolute()
        else [
            repo_root / path,
            repo_root.parent / path,
            kb_root / "pdfs" / path.name,
            kb_root.parent / "pdfs" / path.name,
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    out["title"] = html.unescape(out.get("title") or "").strip().rstrip(".")
    out["abstract"] = html.unescape(out.get("abstract") or "").strip()
    out["venue"] = html.unescape(out.get("venue") or "").strip()
    out["year"] = str(out.get("year") or "").strip()
    out["citationCount"] = int(out.get("citationCount") or 0)
    out.setdefault("externalIds", {})
    return out


def extract_pdf_text(pdf_path: Path) -> ExtractedText:
    import fitz

    document = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in document]
        return ExtractedText(clean_text("\n\n".join(pages)), len(document))
    finally:
        document.close()


def extract_abstract(text: str) -> str:
    match = re.search(
        r"(?is)\babstract\b\s*(.+?)(?:\n\s*(?:1\.?\s*)?introduction\b|\n\s*(?:keywords?|index terms)\b)",
        text[:12000],
    )
    if not match:
        return ""
    abstract = re.sub(r"\s+", " ", match.group(1)).strip(" :-\n\t")
    if 80 <= len(abstract) <= 2500:
        return abstract
    return ""


def split_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("front_matter", [])]

    for line in lines:
        stripped = line.strip()
        section = None
        if len(stripped) <= 80:
            for name, pattern in SECTION_PATTERNS:
                if pattern.match(stripped):
                    section = name
                    break
        if section:
            if sections[-1][1]:
                sections.append((section, []))
            else:
                sections[-1] = (section, [])
            continue
        sections[-1][1].append(line)

    result = []
    for name, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            result.append((name, section_text))
    return result


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size * 0.55:
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_documents(
    kb_root: Path,
    out_root: Path,
    max_papers: int | None,
    force_text: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    repo_root = kb_root.parent
    papers = [normalize_paper(p) for p in load_json(kb_root / "metadata" / "all_papers.json", []) if isinstance(p, dict)]
    download_map = load_json(kb_root / "metadata" / "download_map.json", {})
    if max_papers:
        papers = papers[:max_papers]

    out_text_dir = out_root / "texts"
    out_text_dir.mkdir(parents=True, exist_ok=True)

    enriched_papers = []
    chunks = []
    v2_download_map: dict[str, str] = {}
    by_id = {paper.get("paperId"): paper for paper in papers if paper.get("paperId")}

    total = len(by_id)
    for index, (paper_id, paper) in enumerate(by_id.items(), start=1):
        pdf_path = resolve_path(repo_root, kb_root, download_map.get(paper_id, ""))
        text_path = out_text_dir / f"{safe_paper_id(paper_id)}.txt"
        text = ""
        pages = 0

        if pdf_path:
            v2_download_map[paper_id] = str(pdf_path)
            if text_path.exists() and not force_text:
                text = text_path.read_text(encoding="utf-8", errors="ignore")
            else:
                try:
                    extracted = extract_pdf_text(pdf_path)
                    text = extracted.text
                    pages = extracted.pages
                    if text:
                        text_path.write_text(text, encoding="utf-8")
                except Exception as exc:
                    paper["_v2_extract_error"] = str(exc)

        if text and not paper.get("abstract"):
            abstract = extract_abstract(text)
            if abstract:
                paper["abstract"] = abstract

        paper["_v2_text_path"] = str(text_path) if text_path.exists() else ""
        paper["_v2_text_length"] = len(text)
        paper["_v2_pages"] = pages
        paper["_v2_has_pdf"] = pdf_path is not None
        enriched_papers.append(paper)

        if paper.get("title"):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}::title",
                    "paper_id": paper_id,
                    "chunk_index": 0,
                    "section": "title_abstract",
                    "chunk_type": "title_abstract",
                    "title": paper.get("title", ""),
                    "venue": paper.get("venue", ""),
                    "year": paper.get("year", ""),
                    "text": "\n".join(
                        part
                        for part in [
                            paper.get("title", ""),
                            paper.get("abstract", ""),
                        ]
                        if part
                    ),
                }
            )

        chunk_index = 1
        if text:
            for section, section_text in split_sections(text):
                if section in SKIP_INDEX_SECTIONS:
                    continue
                for chunk in chunk_text(section_text, size=1800, overlap=250):
                    chunks.append(
                        {
                            "chunk_id": f"{paper_id}::c{chunk_index:04d}",
                            "paper_id": paper_id,
                            "chunk_index": chunk_index,
                            "section": section,
                            "chunk_type": "body",
                            "title": paper.get("title", ""),
                            "venue": paper.get("venue", ""),
                            "year": paper.get("year", ""),
                            "text": chunk,
                        }
                    )
                    chunk_index += 1

        if index % 100 == 0 or index == total:
            print(f"documents {index}/{total} chunks={len(chunks)}", flush=True)

    return enriched_papers, chunks, v2_download_map


def write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def build_fts(out_root: Path, papers: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    index_path = out_root / "fts_index.sqlite"
    if index_path.exists():
        index_path.unlink()
    con = sqlite3.connect(index_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            venue TEXT,
            year TEXT,
            abstract TEXT,
            citation_count INTEGER,
            doi TEXT,
            text_path TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            chunk_index INTEGER,
            section TEXT,
            chunk_type TEXT,
            text TEXT
        );
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            paper_id UNINDEXED,
            title,
            venue,
            year,
            section,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    con.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                p.get("paperId"),
                p.get("title", ""),
                p.get("venue", ""),
                p.get("year", ""),
                p.get("abstract", ""),
                int(p.get("citationCount") or 0),
                p.get("doi", ""),
                p.get("_v2_text_path", ""),
            )
            for p in papers
            if p.get("paperId")
        ],
    )
    con.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                c["chunk_id"],
                c["paper_id"],
                c["chunk_index"],
                c["section"],
                c["chunk_type"],
                c["text"],
            )
            for c in chunks
        ],
    )
    con.executemany(
        "INSERT INTO chunk_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c["chunk_id"],
                c["paper_id"],
                c.get("title", ""),
                c.get("venue", ""),
                c.get("year", ""),
                c.get("section", ""),
                c.get("text", ""),
            )
            for c in chunks
        ],
    )
    con.commit()
    con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
    con.close()


def build_vector(out_root: Path, chunks: list[dict[str, Any]], model_name: str, collection_name: str, batch_size: int) -> None:
    import chromadb
    import torch
    from chromadb.utils import embedding_functions

    db_path = out_root / "vector_db"
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, device=device)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
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

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
        print(f"vector {min(end, len(ids))}/{len(ids)}", flush=True)
    print(f"vector_count={collection.count()} device={device}", flush=True)


def write_manifest(out_root: Path, papers: list[dict[str, Any]], chunks: list[dict[str, Any]], started: float) -> None:
    manifest = {
        "version": "v2",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 2),
        "papers": len(papers),
        "papers_with_text": sum(1 for p in papers if p.get("_v2_text_path")),
        "papers_with_abstract": sum(1 for p in papers if p.get("abstract")),
        "chunks": len(chunks),
        "sections": {},
    }
    for chunk in chunks:
        section = chunk.get("section") or "unknown"
        manifest["sections"][section] = manifest["sections"].get(section, 0) + 1
    save_json(out_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ResearchKB v2")
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_KB_ROOT / "v2")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--force-text", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--model-name", default=os.environ.get("RESEARCH_KB_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--collection", default="research_papers_v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    out_root = args.out_root.resolve()
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    papers, chunks, download_map = build_documents(
        kb_root=args.kb_root.resolve(),
        out_root=out_root,
        max_papers=args.max_papers,
        force_text=args.force_text,
    )
    save_json(out_root / "metadata" / "all_papers.json", papers)
    save_json(out_root / "metadata" / "download_map.json", download_map)
    write_chunks(out_root / "chunks.jsonl", chunks)
    build_fts(out_root, papers, chunks)
    if not args.skip_vector:
        build_vector(out_root, chunks, args.model_name, args.collection, args.batch_size)
    write_manifest(out_root, papers, chunks, started)


if __name__ == "__main__":
    main()
