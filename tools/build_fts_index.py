#!/usr/bin/env python3
"""
Build a local SQLite FTS5 index for ResearchKB.

This is a deterministic lexical baseline for paper retrieval. It indexes the
metadata plus extracted text files under research_kb/texts and does not require
Chroma, torch, sentence-transformers, or network access.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_text_lookup(text_dir: Path, paper_ids: set[str]) -> dict[str, Path]:
    by_prefix: dict[str, list[str]] = {}
    for paper_id in paper_ids:
        by_prefix.setdefault(paper_id[:20], []).append(paper_id)

    lookup: dict[str, Path] = {}
    if not text_dir.exists():
        return lookup

    for path in text_dir.glob("*.txt"):
        candidates = by_prefix.get(path.stem, [])
        if len(candidates) == 1:
            lookup[candidates[0]] = path
    return lookup


def read_text(path: Path | None, max_chars: int) -> str:
    if not path:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[:max_chars]


def build_index(kb_root: Path, index_path: Path, max_body_chars: int, rebuild: bool) -> None:
    metadata_path = kb_root / "metadata" / "all_papers.json"
    download_map_path = kb_root / "metadata" / "download_map.json"
    text_dir = kb_root / "texts"

    papers = load_json(metadata_path, [])
    download_map = load_json(download_map_path, {})
    if not isinstance(papers, list):
        raise SystemExit(f"metadata is not a list: {metadata_path}")

    paper_ids = {p.get("paperId") for p in papers if isinstance(p, dict) and p.get("paperId")}
    text_lookup = build_text_lookup(text_dir, paper_ids)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild and index_path.exists():
        index_path.unlink()

    con = sqlite3.connect(index_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            venue TEXT,
            year TEXT,
            abstract TEXT,
            citation_count INTEGER,
            doi TEXT,
            dblp_key TEXT,
            search_keyword TEXT,
            pdf_path TEXT,
            text_path TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS paper_fts USING fts5(
            paper_id UNINDEXED,
            title,
            venue,
            year,
            abstract,
            body,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    con.execute("DELETE FROM papers")
    con.execute("DELETE FROM paper_fts")

    rows = []
    fts_rows = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("paperId")
        if not paper_id:
            continue

        text_path = text_lookup.get(paper_id)
        body = read_text(text_path, max_body_chars)
        pdf_path = download_map.get(paper_id, "")
        rows.append(
            (
                paper_id,
                paper.get("title") or "",
                paper.get("venue") or "",
                str(paper.get("year") or ""),
                paper.get("abstract") or "",
                int(paper.get("citationCount") or 0),
                paper.get("doi") or "",
                paper.get("dblp_key") or "",
                paper.get("_search_keyword") or "",
                pdf_path,
                str(text_path) if text_path else "",
            )
        )
        fts_rows.append(
            (
                paper_id,
                paper.get("title") or "",
                paper.get("venue") or "",
                str(paper.get("year") or ""),
                paper.get("abstract") or "",
                body,
            )
        )

    con.executemany(
        """
        INSERT INTO papers (
            paper_id, title, venue, year, abstract, citation_count, doi,
            dblp_key, search_keyword, pdf_path, text_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.executemany(
        "INSERT INTO paper_fts (paper_id, title, venue, year, abstract, body) VALUES (?, ?, ?, ?, ?, ?)",
        fts_rows,
    )
    con.commit()
    con.execute("INSERT INTO paper_fts(paper_fts) VALUES('optimize')")
    con.close()

    with_text = sum(1 for row in rows if row[-1])
    print(f"indexed_papers={len(rows)}")
    print(f"indexed_texts={with_text}")
    print(f"index_path={index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ResearchKB SQLite FTS5 index")
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_KB_ROOT / "fts_index.sqlite")
    parser.add_argument("--max-body-chars", type=int, default=120_000)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    build_index(
        kb_root=args.kb_root.resolve(),
        index_path=args.index.resolve(),
        max_body_chars=args.max_body_chars,
        rebuild=not args.no_rebuild,
    )


if __name__ == "__main__":
    main()
