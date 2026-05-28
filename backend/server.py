"""
ResearchDB API
==============
面向其他服务器/服务调用的论文知识库后端。

能力:
  1. 向量检索 research_kb 中的论文
  2. 返回论文元数据
  3. 返回本地 PDF 全文清洗文本
  4. 直出本地缓存 PDF
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = Path(os.environ.get("RESEARCH_KB_ROOT", REPO_ROOT / "research_kb")).resolve()
KB_V2_ROOT = Path(os.environ.get("RESEARCH_KB_V2_ROOT", KB_ROOT / "v2")).resolve()
DEFAULT_INDEX_ROOT = KB_V2_ROOT if (KB_V2_ROOT / "metadata" / "all_papers.json").exists() else KB_ROOT
METADATA_PATH = Path(
    os.environ.get("RESEARCH_KB_METADATA_PATH", DEFAULT_INDEX_ROOT / "metadata" / "all_papers.json")
).resolve()
DOWNLOAD_MAP_PATH = Path(
    os.environ.get("RESEARCH_KB_DOWNLOAD_MAP_PATH", DEFAULT_INDEX_ROOT / "metadata" / "download_map.json")
).resolve()
VECTOR_DB_PATH = Path(
    os.environ.get("RESEARCH_KB_VECTOR_DB_PATH", DEFAULT_INDEX_ROOT / "vector_db")
).resolve()
FTS_INDEX_PATH = Path(
    os.environ.get("RESEARCH_KB_FTS_INDEX_PATH", DEFAULT_INDEX_ROOT / "fts_index.sqlite")
).resolve()
PDF_ROOT = Path(os.environ.get("RESEARCH_KB_PDF_ROOT", KB_ROOT / "pdfs")).resolve()
TEXT_CACHE_DIR = Path(
    os.environ.get("RESEARCH_KB_TEXT_CACHE_DIR", DEFAULT_INDEX_ROOT / "texts")
).resolve()

MODEL_NAME = os.environ.get("RESEARCH_KB_EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.environ.get(
    "RESEARCH_KB_COLLECTION",
    "research_papers_v2" if DEFAULT_INDEX_ROOT == KB_V2_ROOT else "research_papers",
)

VENUE_ALIASES = {
    "aaai": "AAAI",
    "cvpr": "CVPR",
    "iccv": "ICCV",
    "eccv": "ECCV",
    "wacv": "WACV",
    "neurips": "NEURIPS",
    "nips": "NEURIPS",
    "iclr": "ICLR",
    "icml": "ICML",
}

CHINESE_QUERY_EXPANSIONS = {
    "遥感": "remote sensing",
    "语义分割": "semantic segmentation",
    "实例分割": "instance segmentation",
    "图像分割": "image segmentation",
    "变化检测": "change detection",
    "目标检测": "object detection",
    "多模态": "multimodal",
    "大模型": "large language model foundation model",
}


def resolve_embedding_model(model_name: str) -> str:
    """优先使用本地缓存的 HF snapshot，避免服务运行时再访问网络。"""
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if "/" not in model_name:
        return model_name

    org, repo = model_name.split("/", 1)
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}"
    if not hub_dir.exists():
        return model_name

    ref_main = hub_dir / "refs" / "main"
    if ref_main.exists():
        snapshot_id = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = hub_dir / "snapshots" / snapshot_id
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())

    snapshots_dir = hub_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            [p for p in snapshots_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0].resolve())

    return model_name


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"<latexit[^>]*>.*?</latexit>", "", text, flags=re.DOTALL)
    text = re.sub(r"[A-Za-z0-9+/=]{50,}", "", text)
    text = re.sub(r"[^\x00-\x7F\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]{5,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def safe_filename(name: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.")
    return value or fallback


def parse_search_constraints(query: str) -> dict[str, Any]:
    compact = re.sub(r"[\s_\-]+", "", query.lower())
    years = sorted(set(re.findall(r"(?:19|20)\d{2}", query)))
    venues = []
    for alias, canonical in VENUE_ALIASES.items():
        if alias in compact and canonical not in venues:
            venues.append(canonical)
    return {"venues": venues, "years": years}


def strip_search_constraints(query: str, constraints: dict[str, Any]) -> str:
    cleaned = query
    for alias in VENUE_ALIASES:
        cleaned = re.sub(rf"(?i){re.escape(alias)}[\s_-]*(?:19|20)\d{{2}}", " ", cleaned)
        cleaned = re.sub(rf"(?i)\b{re.escape(alias)}\b", " ", cleaned)
    for year in constraints.get("years", []):
        cleaned = cleaned.replace(str(year), " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query


def expand_query_terms(query: str) -> str:
    expansions = [value for key, value in CHINESE_QUERY_EXPANSIONS.items() if key in query]
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"


def retrieval_query(query: str, constraints: dict[str, Any]) -> str:
    return expand_query_terms(strip_search_constraints(query, constraints))


def normalized_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def matches_search_constraints(item: dict[str, Any], constraints: dict[str, Any]) -> bool:
    years = {str(year) for year in constraints.get("years", [])}
    if years and str(item.get("year") or "") not in years:
        return False
    venues = constraints.get("venues", [])
    if venues:
        venue_text = normalized_text(item.get("venue") or item.get("source_conference") or "")
        if not any(normalized_text(venue) in venue_text for venue in venues):
            return False
    return True


class ResearchKBStore:
    def __init__(self):
        self.refresh()

    def refresh(self):
        self.papers = load_json(METADATA_PATH, [])
        self.download_map = load_json(DOWNLOAD_MAP_PATH, {})
        self.paper_by_id = {
            paper["paperId"]: paper for paper in self.papers if isinstance(paper, dict) and paper.get("paperId")
        }

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        paper = self.paper_by_id.get(paper_id)
        if not paper:
            raise KeyError(paper_id)
        return paper

    def resolve_pdf_path(self, paper_id: str) -> Optional[Path]:
        relative = self.download_map.get(paper_id)
        if not relative:
            return None

        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate

        fallback = (PDF_ROOT / Path(relative).name).resolve()
        if fallback.exists():
            return fallback
        return None

    def text_cache_path(self, paper_id: str) -> Path:
        TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return TEXT_CACHE_DIR / f"{paper_id}.txt"

    def build_paper_payload(self, paper: dict[str, Any]) -> dict[str, Any]:
        paper_id = paper["paperId"]
        pdf_path = self.resolve_pdf_path(paper_id)
        open_access_pdf = paper.get("openAccessPdf") or {}
        return {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year", ""),
            "abstract": paper.get("abstract", ""),
            "citation_count": paper.get("citationCount", 0),
            "doi": paper.get("doi", ""),
            "dblp_key": paper.get("dblp_key", ""),
            "external_ids": paper.get("externalIds", {}),
            "authors": paper.get("authors", []),
            "pages": paper.get("pages", ""),
            "source_conference": paper.get("_source_conference", ""),
            "source_issue": paper.get("_source_issue", ""),
            "source_url": paper.get("_source_url", ""),
            "search_keyword": paper.get("_search_keyword", ""),
            "open_access_pdf_url": open_access_pdf.get("url", ""),
            "pdf_available": pdf_path is not None,
            "fulltext_cached": self.text_cache_path(paper_id).exists(),
        }

    def keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = [token for token in re.split(r"\s+", query.lower().strip()) if token]
        if not terms:
            return []

        scored = []
        for paper in self.papers:
            title = (paper.get("title") or "").lower()
            abstract = (paper.get("abstract") or "").lower()
            venue = (paper.get("venue") or "").lower()

            score = 0
            for term in terms:
                if term in title:
                    score += 5
                if term in abstract:
                    score += 2
                if term in venue:
                    score += 1

            if score <= 0:
                continue

            scored.append(
                {
                    "score": score,
                    "paper": paper,
                    "snippet": (paper.get("abstract") or "")[:800],
                }
            )

        scored.sort(
            key=lambda item: (
                -item["score"],
                -(item["paper"].get("citationCount") or 0),
                str(item["paper"].get("year") or ""),
            )
        )
        return scored[:limit]


store = ResearchKBStore()


@lru_cache(maxsize=1)
def get_vector_collection():
    try:
        import chromadb
        import torch
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise RuntimeError(
            "缺少向量检索依赖，请安装 chromadb sentence-transformers torch"
        ) from exc

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = resolve_embedding_model(MODEL_NAME)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        device=device,
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)


def search_vector_db(query: str, limit: int, constraints: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    collection = get_vector_collection()
    constraints = constraints or {}
    fetch_limit = max(limit * (12 if constraints.get("venues") or constraints.get("years") else 3), 80 if constraints else limit)
    result = collection.query(query_texts=[retrieval_query(query, constraints)], n_results=fetch_limit)

    matches = []
    for idx in range(len(result["ids"][0])):
        metadata = result["metadatas"][0][idx] or {}
        paper_id = metadata.get("paper_id", "")
        paper = store.paper_by_id.get(paper_id, {})

        if paper_id and paper:
            base = store.build_paper_payload(paper)
        else:
            base = {
                "paper_id": paper_id,
                "title": metadata.get("title", ""),
                "venue": metadata.get("venue", ""),
                "year": metadata.get("year", ""),
                "abstract": "",
                "citation_count": metadata.get("citations", 0),
                "doi": "",
                "dblp_key": "",
                "external_ids": {},
                "source_conference": "",
                "search_keyword": "",
                "open_access_pdf_url": "",
                "pdf_available": False,
                "fulltext_cached": False,
            }

        base.update(
            {
                "chunk_id": result["ids"][0][idx],
                "score": result["distances"][0][idx] if result.get("distances") else None,
                "snippet": (result["documents"][0][idx] or "")[:800],
                "section": metadata.get("section", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "retrieval_source": "vector",
            }
        )
        matches.append(base)

    seen_papers = set()
    deduped = []
    for item in matches:
        if constraints and not matches_search_constraints(item, constraints):
            continue
        paper_key = item.get("paper_id") or item.get("title", "")
        if paper_key and paper_key in seen_papers:
            continue
        if paper_key:
            seen_papers.add(paper_key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def fts_query_string(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query.lower())
    expanded = []
    for token in tokens:
        if "-" in token:
            expanded.extend(part for part in token.split("-") if part)
        expanded.append(token.replace('"', '""'))
    return " OR ".join(f'"{token}"' for token in expanded if token)


def fts_filter_sql(constraints: dict[str, Any]) -> tuple[str, list[Any]]:
    conditions = []
    params: list[Any] = []
    years = constraints.get("years", [])
    venues = constraints.get("venues", [])
    if years:
        conditions.append(f"p.year IN ({','.join('?' for _ in years)})")
        params.extend(str(year) for year in years)
    if venues:
        conditions.append("(" + " OR ".join("UPPER(p.venue) LIKE ?" for _ in venues) + ")")
        params.extend(f"%{venue.upper()}%" for venue in venues)
    if not conditions:
        return "", []
    return " AND " + " AND ".join(conditions), params


def search_fts_db(query: str, limit: int, constraints: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    if not FTS_INDEX_PATH.exists():
        raise RuntimeError(f"FTS index not found: {FTS_INDEX_PATH}")

    constraints = constraints or {}
    fts_query = fts_query_string(retrieval_query(query, constraints))
    if not fts_query and not constraints:
        return []
    filter_sql, filter_params = fts_filter_sql(constraints)

    connection = sqlite3.connect(FTS_INDEX_PATH)
    connection.row_factory = sqlite3.Row
    try:
        has_chunk_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
        ).fetchone()
        if has_chunk_fts and fts_query:
            rows = connection.execute(
                f"""
                SELECT
                    p.paper_id,
                    p.title,
                    p.venue,
                    p.year,
                    p.citation_count,
                    c.chunk_id,
                    c.section,
                    c.chunk_type,
                    bm25(chunk_fts, 8.0, 2.5, 1.0, 1.0, 3.0, 1.0) AS score,
                    snippet(chunk_fts, 6, '', '', ' ... ', 120) AS snippet
                FROM chunk_fts
                JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
                JOIN papers p ON p.paper_id = chunk_fts.paper_id
                WHERE chunk_fts MATCH ? {filter_sql}
                ORDER BY score ASC, CAST(p.year AS INTEGER) DESC, p.citation_count DESC
                LIMIT ?
                """,
                (fts_query, *filter_params, max(limit * 6, limit)),
            ).fetchall()
        elif fts_query:
            rows = connection.execute(
                f"""
                SELECT
                    p.paper_id,
                    p.title,
                    p.venue,
                    p.year,
                    p.citation_count,
                    NULL AS chunk_id,
                    '' AS section,
                    '' AS chunk_type,
                    bm25(paper_fts, 8.0, 2.5, 1.0, 4.0, 1.0) AS score,
                    snippet(paper_fts, 5, '', '', ' ... ', 120) AS snippet
                FROM paper_fts
                JOIN papers p ON p.paper_id = paper_fts.paper_id
                WHERE paper_fts MATCH ? {filter_sql}
                ORDER BY score ASC, CAST(p.year AS INTEGER) DESC, p.citation_count DESC
                LIMIT ?
                """,
                (fts_query, *filter_params, max(limit * 6, limit)),
            ).fetchall()
        else:
            rows = connection.execute(
                f"""
                SELECT
                    p.paper_id,
                    p.title,
                    p.venue,
                    p.year,
                    p.citation_count,
                    NULL AS chunk_id,
                    '' AS section,
                    '' AS chunk_type,
                    0.0 AS score,
                    p.abstract AS snippet
                FROM papers p
                WHERE 1=1 {filter_sql}
                ORDER BY CAST(p.year AS INTEGER) DESC, p.citation_count DESC
                LIMIT ?
                """,
                (*filter_params, max(limit * 6, limit)),
            ).fetchall()
    finally:
        connection.close()

    matches = []
    seen_papers = set()
    for row in rows:
        paper_id = row["paper_id"]
        if paper_id in seen_papers:
            continue
        seen_papers.add(paper_id)
        paper = store.paper_by_id.get(paper_id)
        if paper:
            base = store.build_paper_payload(paper)
        else:
            base = {
                "paper_id": paper_id,
                "title": row["title"],
                "venue": row["venue"],
                "year": row["year"],
                "abstract": "",
                "citation_count": row["citation_count"],
                "doi": "",
                "dblp_key": "",
                "external_ids": {},
                "authors": [],
                "pages": "",
                "source_conference": "",
                "source_issue": "",
                "source_url": "",
                "search_keyword": "",
                "open_access_pdf_url": "",
                "pdf_available": False,
                "fulltext_cached": False,
            }
        base.update(
            {
                "chunk_id": row["chunk_id"],
                "score": row["score"],
                "snippet": row["snippet"] or "",
                "section": row["section"] or "",
                "chunk_type": row["chunk_type"] or "",
                "retrieval_source": "fts",
            }
        )
        matches.append(base)
        if len(matches) >= limit:
            break
    return matches


def lexical_signal(query: str, item: dict[str, Any]) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not tokens:
        return 0
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    signal = 0
    for token in tokens:
        if token in title:
            signal += 3
        if token in snippet:
            signal += 1
    return signal


def search_hybrid_db(query: str, limit: int) -> list[dict[str, Any]]:
    constraints = parse_search_constraints(query)
    candidates: dict[str, dict[str, Any]] = {}
    errors = []

    for source, fetch_limit, fetcher in (
        ("vector", max(limit * 4, 20), search_vector_db),
        ("fts", max(limit * 4, 20), search_fts_db),
    ):
        try:
            results = fetcher(query, fetch_limit, constraints)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            continue

        for rank, item in enumerate(results, start=1):
            paper_id = item.get("paper_id") or item.get("title") or ""
            if not paper_id:
                continue
            current = candidates.get(paper_id)
            if not current:
                current = dict(item)
                current["retrieval_sources"] = []
                current["_hybrid_score"] = 0.0
                candidates[paper_id] = current
            current["retrieval_sources"].append(source)
            current["_hybrid_score"] += 1.0 / (rank + 20)
            if not current.get("snippet") and item.get("snippet"):
                current["snippet"] = item["snippet"]

    if not candidates and errors:
        raise RuntimeError("; ".join(errors))

    items = list(candidates.values())
    for item in items:
        if constraints and not matches_search_constraints(item, constraints):
            item["_hybrid_score"] -= 10
        item["_hybrid_score"] += 0.01 * lexical_signal(retrieval_query(query, constraints), item)
        if constraints.get("venues") or constraints.get("years"):
            item["_hybrid_score"] += 0.5
        try:
            item["_year_sort"] = int(item.get("year") or 0)
        except ValueError:
            item["_year_sort"] = 0
        item["retrieval_source"] = "+".join(sorted(set(item.get("retrieval_sources", []))))
        item["score"] = round(item["_hybrid_score"], 6)

    items.sort(
        key=lambda item: (
            -item["_hybrid_score"],
            -item["_year_sort"],
            -(item.get("citation_count") or 0),
            item.get("title") or "",
        )
    )
    for item in items:
        item.pop("_hybrid_score", None)
        item.pop("_year_sort", None)
        item["parsed_filters"] = constraints
    return items[:limit]


def get_pdf_path_or_404(paper_id: str) -> Path:
    try:
        store.get_paper(paper_id)
    except KeyError:
        raise HTTPException(404, "paper_id 不存在") from None

    pdf_path = store.resolve_pdf_path(paper_id)
    if not pdf_path:
        raise HTTPException(404, "该论文未找到本地 PDF")
    return pdf_path


def read_fulltext(paper_id: str, max_chars: int, force_refresh: bool) -> dict[str, Any]:
    pdf_path = get_pdf_path_or_404(paper_id)
    cache_path = store.text_cache_path(paper_id)

    if cache_path.exists() and not force_refresh:
        text = cache_path.read_text(encoding="utf-8")
        return {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "cache_path": str(cache_path),
            "cached": True,
            "text": text[:max_chars],
            "text_length": min(len(text), max_chars),
        }

    try:
        import fitz
    except ImportError as exc:
        raise HTTPException(503, "缺少 PyMuPDF，请安装 pymupdf") from exc

    try:
        document = fitz.open(pdf_path)
        raw_text = "\n".join(page.get_text() for page in document)
        page_count = len(document)
        document.close()
    except Exception as exc:
        raise HTTPException(500, f"PDF 读取失败: {exc}") from exc

    cleaned = clean_text(raw_text)
    cache_path.write_text(cleaned, encoding="utf-8")
    return {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "cache_path": str(cache_path),
        "cached": False,
        "pages": page_count,
        "text": cleaned[:max_chars],
        "text_length": min(len(cleaned), max_chars),
    }


app = FastAPI(title="ResearchDB API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from research_kb.backend.reader import (
        app as reader_app,
        init_db as init_reader_db,
        mark_stale_llm_jobs_failed,
        start_auto_translation_worker,
    )
except Exception as exc:  # pragma: no cover - keeps ResearchDB available if reader deps break.
    reader_app = None
    init_reader_db = None
    mark_stale_llm_jobs_failed = None
    start_auto_translation_worker = None
    log.exception("Paper Reader backend could not be mounted: %s", exc)
else:
    app.mount("/reader", reader_app)
    log.info("Paper Reader backend mounted at /reader")


@app.on_event("startup")
def startup():
    if init_reader_db is None:
        return
    init_reader_db()
    mark_stale_llm_jobs_failed()
    start_auto_translation_worker()
    log.info("Paper Reader backend initialized inside ResearchDB service")


@app.get("/")
def root():
    return {
        "service": "researchdb",
        "docs": "/docs",
        "health": "/healthz",
        "reader": "/reader/api/stats" if reader_app is not None else None,
        "papers": len(store.papers),
    }


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "researchdb",
        "time": datetime.now(timezone.utc).isoformat(),
        "kb_root": str(KB_ROOT),
        "index_root": str(DEFAULT_INDEX_ROOT),
        "metadata_exists": METADATA_PATH.exists(),
        "download_map_exists": DOWNLOAD_MAP_PATH.exists(),
        "vector_db_exists": VECTOR_DB_PATH.exists(),
        "fts_index_exists": FTS_INDEX_PATH.exists(),
        "papers": len(store.papers),
        "pdf_mapped": len(store.download_map),
        "collection": COLLECTION_NAME,
        "reader_mounted": reader_app is not None,
    }


@app.post("/api/admin/reload")
def reload_metadata():
    get_vector_collection.cache_clear()
    store.refresh()
    return {
        "ok": True,
        "papers": len(store.papers),
        "pdf_mapped": len(store.download_map),
    }


@app.get("/api/stats")
def stats():
    cached_texts = len(list(TEXT_CACHE_DIR.glob("*.txt"))) if TEXT_CACHE_DIR.exists() else 0
    by_venue = {}
    for paper in store.papers:
        venue = paper.get("venue") or "UNKNOWN"
        by_venue[venue] = by_venue.get(venue, 0) + 1
    return {
        "papers": len(store.papers),
        "pdf_mapped": len(store.download_map),
        "text_cached": cached_texts,
        "venues": by_venue,
    }


@app.get("/api/search")
def search(
    query: str = Query(..., min_length=1, description="建议英文关键词，效果最好"),
    limit: int = Query(10, ge=1, le=50),
    mode: str = Query("auto", pattern="^(auto|vector|keyword|fts|hybrid)$"),
):
    constraints = parse_search_constraints(query)
    if mode == "hybrid":
        try:
            items = search_hybrid_db(query, limit)
            return {
                "query": query,
                "mode": "hybrid",
                "parsed_filters": constraints,
                "count": len(items),
                "items": items,
            }
        except Exception as exc:
            raise HTTPException(503, f"混合检索不可用: {exc}") from exc

    if mode == "fts":
        try:
            items = search_fts_db(query, limit, constraints)
            return {
                "query": query,
                "mode": "fts",
                "parsed_filters": constraints,
                "count": len(items),
                "items": items,
            }
        except Exception as exc:
            raise HTTPException(503, f"FTS 检索不可用: {exc}") from exc

    if mode in {"auto", "vector"}:
        try:
            items = search_vector_db(query, limit, constraints)
            return {
                "query": query,
                "mode": "vector",
                "parsed_filters": constraints,
                "count": len(items),
                "items": items,
            }
        except Exception as exc:
            if mode == "vector":
                raise HTTPException(503, f"向量检索不可用: {exc}") from exc

    fallback = []
    for item in store.keyword_search(retrieval_query(query, constraints), limit * 4):
        payload = store.build_paper_payload(item["paper"])
        if constraints and not matches_search_constraints(payload, constraints):
            continue
        payload.update(
            {
                "chunk_id": None,
                "score": item["score"],
                "snippet": item["snippet"],
            }
        )
        fallback.append(payload)
        if len(fallback) >= limit:
            break

    return {
        "query": query,
        "mode": "keyword",
        "parsed_filters": constraints,
        "count": len(fallback),
        "items": fallback,
    }


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str):
    try:
        paper = store.get_paper(paper_id)
    except KeyError:
        raise HTTPException(404, "paper_id 不存在") from None
    return store.build_paper_payload(paper)


@app.get("/api/papers/{paper_id}/fulltext")
def get_fulltext(
    paper_id: str,
    max_chars: int = Query(120000, ge=1000, le=1_000_000),
    force_refresh: bool = False,
):
    payload = read_fulltext(paper_id, max_chars=max_chars, force_refresh=force_refresh)
    paper = store.get_paper(paper_id)
    payload.update(
        {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year", ""),
        }
    )
    return payload


@app.get("/api/papers/{paper_id}/pdf")
def get_pdf(paper_id: str):
    pdf_path = get_pdf_path_or_404(paper_id)
    paper = store.get_paper(paper_id)
    filename = safe_filename(paper.get("title", ""), paper_id)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{filename}.pdf")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18000)
