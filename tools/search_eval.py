#!/usr/bin/env python3
"""
Compare ResearchKB retrieval modes on a small, explicit query set.

The script is intentionally lightweight: it can call the running ResearchDB API
for vector/keyword results and can also query the local SQLite FTS5 index built
by build_fts_index.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "http://10.70.199.159:18000"


@dataclass(frozen=True)
class EvalQuery:
    name: str
    query: str
    must_terms: tuple[str, ...]
    nice_terms: tuple[str, ...] = ()


QUERIES = [
    EvalQuery(
        name="remote_sensing_change",
        query="remote sensing change detection",
        must_terms=("remote", "change"),
        nice_terms=("sensing", "satellite", "temporal"),
    ),
    EvalQuery(
        name="open_vocab_segmentation",
        query="open vocabulary semantic segmentation",
        must_terms=("open", "segmentation"),
        nice_terms=("vocabulary", "semantic", "clip"),
    ),
    EvalQuery(
        name="ptq_vit",
        query="post-training quantization vision transformer",
        must_terms=("quantization", "transformer"),
        nice_terms=("post-training", "ptq", "vit"),
    ),
    EvalQuery(
        name="image_registration",
        query="image registration deep learning",
        must_terms=("registration",),
        nice_terms=("image", "deformable", "matching"),
    ),
    EvalQuery(
        name="sam_medical",
        query="segment anything medical image segmentation",
        must_terms=("segment", "medical"),
        nice_terms=("sam", "anything", "segmentation"),
    ),
]


def api_get_json(url: str, timeout: int = 45) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_api(api_base: str, query: str, mode: str, limit: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"query": query, "limit": limit, "mode": mode})
    payload = api_get_json(f"{api_base.rstrip('/')}/api/search?{params}")
    return payload.get("items", [])


def fts_query_string(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query.lower())
    expanded: list[str] = []
    for token in tokens:
        if "-" in token:
            expanded.extend(part for part in token.split("-") if part)
        expanded.append(token.replace('"', '""'))
    return " OR ".join(f'"{token}"' for token in expanded if token)


def search_fts(index_path: Path, query: str, limit: int) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    fts_query = fts_query_string(query)
    if not fts_query:
        return []
    con = sqlite3.connect(index_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT
            p.paper_id,
            p.title,
            p.venue,
            p.year,
            p.citation_count,
            p.text_path,
            bm25(paper_fts, 8.0, 2.5, 1.0, 4.0, 1.0) AS score,
            snippet(paper_fts, 5, '[', ']', ' ... ', 80) AS snippet
        FROM paper_fts
        JOIN papers p ON p.paper_id = paper_fts.paper_id
        WHERE paper_fts MATCH ?
        ORDER BY score ASC, CAST(p.year AS INTEGER) DESC, p.citation_count DESC
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    con.close()
    return [dict(row) for row in rows]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def heuristic_relevance(item: dict[str, Any], query: EvalQuery) -> tuple[int, str]:
    haystack = normalize_text(
        " ".join(
            str(item.get(key) or "")
            for key in ("title", "venue", "year", "abstract", "snippet")
        )
    )
    must_hits = [term for term in query.must_terms if term.lower() in haystack]
    nice_hits = [term for term in query.nice_terms if term.lower() in haystack]
    score = len(must_hits) * 2 + len(nice_hits)
    reason = f"must={len(must_hits)}/{len(query.must_terms)} nice={len(nice_hits)}/{len(query.nice_terms)}"
    return score, reason


def summarize_method(name: str, query: EvalQuery, items: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    annotated = []
    seen_papers = set()
    for rank, item in enumerate(items[:top_k], start=1):
        relevance, reason = heuristic_relevance(item, query)
        paper_id = item.get("paper_id") or item.get("paperId") or ""
        seen_papers.add(paper_id)
        annotated.append(
            {
                "rank": rank,
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "venue": item.get("venue", ""),
                "year": item.get("year", ""),
                "score": item.get("score"),
                "heuristic": relevance,
                "reason": reason,
                "snippet": normalize_text(str(item.get("snippet") or ""))[:280],
            }
        )
    avg = sum(x["heuristic"] for x in annotated) / len(annotated) if annotated else 0.0
    return {
        "method": name,
        "count": len(items),
        "unique_papers": len(seen_papers),
        "avg_heuristic_topk": round(avg, 3),
        "top": annotated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResearchKB search quality")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--index", type=Path, default=DEFAULT_KB_ROOT / "fts_index.sqlite")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["vector", "keyword", "fts"])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: list[dict[str, Any]] = []
    for query in QUERIES:
        query_report = {"name": query.name, "query": query.query, "methods": []}
        print(f"\n## {query.name}: {query.query}")
        for method in args.methods:
            try:
                if method in {"vector", "keyword", "auto", "hybrid"}:
                    items = search_api(args.api_base, query.query, method, args.limit)
                elif method == "fts":
                    items = search_fts(args.index, query.query, args.limit)
                else:
                    raise ValueError(f"unknown method: {method}")
                summary = summarize_method(method, query, items, args.top_k)
            except Exception as exc:
                summary = {"method": method, "error": str(exc), "count": 0, "top": []}
            query_report["methods"].append(summary)

            if "error" in summary:
                print(f"- {method}: ERROR {summary['error']}")
                continue
            print(
                f"- {method}: count={summary['count']} "
                f"avg_h={summary['avg_heuristic_topk']} unique={summary['unique_papers']}"
            )
            for item in summary["top"][:3]:
                print(f"  {item['rank']}. {item['title']} | {item['venue']} {item['year']} | {item['reason']}")
        report.append(query_report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
