#!/usr/bin/env python3
"""Import ResearchDB metadata into the reader SQLite database."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = REPO_ROOT / "research_kb"
DEFAULT_METADATA = KB_ROOT / "v2" / "metadata" / "all_papers.json"
DEFAULT_READER_DB = Path(
    os.environ.get("READER_DATA_DIR", KB_ROOT / "reader_data")
) / "papers.db"
RESEARCH_ID_PREFIXES = (
    "aaai_",
    "arxiv_",
    "cvf_",
    "dblp_",
    "openreview_",
    "semanticscholar_",
    "s2_",
)


def normalize_conference(value: str) -> str:
    text = re.sub(r"\s+", "", value.strip().upper())
    return text or "RESEARCHDB"


def title_key(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "[]", "{}", "null", "None"}


def is_research_id(paper_id: str) -> bool:
    return paper_id.startswith(RESEARCH_ID_PREFIXES)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paper_row(item: dict) -> dict:
    paper_id = str(item.get("paperId") or "").strip()
    venue = str(item.get("venue") or item.get("_source_conference") or "").strip()
    open_access_pdf = item.get("openAccessPdf") or {}
    pdf_url = str(open_access_pdf.get("url") or "").strip()
    source_url = str(item.get("_source_url") or "").strip()
    authors = item.get("authors") if isinstance(item.get("authors"), list) else []
    pages = str(item.get("pages") or item.get("_v2_pages") or "").strip()
    return {
        "id": paper_id,
        "conference": normalize_conference(venue),
        "title": str(item.get("title") or paper_id).strip(),
        "authors": json.dumps(authors, ensure_ascii=False),
        "abstract": str(item.get("abstract") or "").strip(),
        "pdf_url": pdf_url,
        "page_url": source_url,
        "pages": pages,
    }


def import_papers(metadata_path: Path, reader_db: Path) -> dict:
    items = load_json(metadata_path)
    reader_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(reader_db))
    conn.row_factory = sqlite3.Row
    try:
        existing_ids = {
            row[0]
            for row in conn.execute("SELECT id FROM papers").fetchall()
        }
        existing_title_ids = {
            (row["conference"], title_key(row["title"])): row["id"]
            for row in conn.execute("SELECT id, conference, title FROM papers").fetchall()
        }
        inserted = 0
        updated = 0
        skipped = 0
        matched_existing = 0
        for item in items:
            row = paper_row(item)
            if not row["id"]:
                skipped += 1
                continue
            existing_id = existing_title_ids.get((row["conference"], title_key(row["title"])))
            if existing_id and existing_id != row["id"]:
                row["id"] = existing_id
                matched_existing += 1
            if row["id"] in existing_ids:
                updated += 1
            else:
                inserted += 1
                existing_ids.add(row["id"])
            existing_title_ids[(row["conference"], title_key(row["title"]))] = row["id"]
            conn.execute(
                """
                INSERT INTO papers
                    (id, conference, title, authors, abstract, pdf_url, page_url, pages)
                VALUES
                    (:id, :conference, :title, :authors, :abstract, :pdf_url, :page_url, :pages)
                ON CONFLICT(id) DO UPDATE SET
                    conference=excluded.conference,
                    title=excluded.title,
                    authors=excluded.authors,
                    abstract=excluded.abstract,
                    pdf_url=excluded.pdf_url,
                    page_url=excluded.page_url,
                    pages=excluded.pages
                """,
                row,
            )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return {
            "metadata": str(metadata_path),
            "reader_db": str(reader_db),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "matched_existing": matched_existing,
            "total": total,
        }
    finally:
        conn.close()


def backup_database(reader_db: Path) -> Path:
    backup = reader_db.with_suffix(reader_db.suffix + ".bak")
    index = 1
    while backup.exists():
        backup = reader_db.with_suffix(reader_db.suffix + f".bak.{index}")
        index += 1
    shutil.copy2(reader_db, backup)
    return backup


def state_counts(conn: sqlite3.Connection, paper_ids: list[str]) -> dict[str, int]:
    counts = {paper_id: 0 for paper_id in paper_ids}
    if not paper_ids:
        return counts
    placeholders = ",".join("?" for _ in paper_ids)
    for table in (
        "analyses",
        "reading_status",
        "notes",
        "translation_chunks",
        "figure_refs",
        "llm_jobs",
    ):
        for row in conn.execute(
            f"""
            SELECT paper_id, COUNT(*) AS count
            FROM {table}
            WHERE paper_id IN ({placeholders})
            GROUP BY paper_id
            """,
            paper_ids,
        ):
            counts[row["paper_id"]] += int(row["count"])
    return counts


def choose_canonical(rows: list[sqlite3.Row], counts: dict[str, int]) -> sqlite3.Row:
    def score(row: sqlite3.Row) -> tuple[int, int, str]:
        paper_id = row["id"]
        reader_id_score = 1 if not is_research_id(paper_id) else 0
        created_at = row["created_at"] or ""
        return (reader_id_score, counts.get(paper_id, 0), created_at)

    return sorted(rows, key=score, reverse=True)[0]


def merge_paper_metadata(conn: sqlite3.Connection, canonical: sqlite3.Row, duplicate: sqlite3.Row) -> None:
    updates: dict[str, object] = {}
    for column in ("authors", "abstract", "pdf_url", "page_url", "arxiv_url", "pages"):
        current = canonical[column]
        candidate = duplicate[column]
        if is_blank(current) and not is_blank(candidate):
            updates[column] = candidate
    if updates:
        assignments = ", ".join(f"{column}=:{column}" for column in updates)
        updates["id"] = canonical["id"]
        conn.execute(f"UPDATE papers SET {assignments} WHERE id=:id", updates)


def merge_analyses(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    source = conn.execute("SELECT * FROM analyses WHERE paper_id=?", (source_id,)).fetchone()
    if not source:
        return
    target = conn.execute("SELECT * FROM analyses WHERE paper_id=?", (target_id,)).fetchone()
    if not target:
        conn.execute("UPDATE analyses SET paper_id=? WHERE paper_id=?", (target_id, source_id))
        return

    updates: dict[str, object] = {"paper_id": target_id}
    for column in ("sections_json", "analysis", "model", "translation", "translation_model"):
        if is_blank(target[column]) and not is_blank(source[column]):
            updates[column] = source[column]
    for column in ("token_count", "translation_token_count"):
        if int(target[column] or 0) <= 0 and int(source[column] or 0) > 0:
            updates[column] = source[column]
    if is_blank(target["translation_created_at"]) and not is_blank(source["translation_created_at"]):
        updates["translation_created_at"] = source["translation_created_at"]
    if len(updates) > 1:
        assignments = ", ".join(f"{column}=:{column}" for column in updates if column != "paper_id")
        conn.execute(f"UPDATE analyses SET {assignments} WHERE paper_id=:paper_id", updates)
    conn.execute("DELETE FROM analyses WHERE paper_id=?", (source_id,))


def merge_reading_status(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    source = conn.execute("SELECT * FROM reading_status WHERE paper_id=?", (source_id,)).fetchone()
    if not source:
        return
    target = conn.execute("SELECT * FROM reading_status WHERE paper_id=?", (target_id,)).fetchone()
    if not target:
        conn.execute("UPDATE reading_status SET paper_id=? WHERE paper_id=?", (target_id, source_id))
        return

    priority = {"unread": 0, "reading": 1, "read": 2}
    source_progress = float(source["progress"] or 0)
    target_progress = float(target["progress"] or 0)
    source_better = (
        priority.get(source["status"], 0) > priority.get(target["status"], 0)
        or source_progress > target_progress
    )
    if source_better:
        conn.execute(
            """
            UPDATE reading_status
            SET status=?, progress=?, updated_at=?
            WHERE paper_id=?
            """,
            (source["status"], source["progress"], source["updated_at"], target_id),
        )
    conn.execute("DELETE FROM reading_status WHERE paper_id=?", (source_id,))


def merge_translation_chunks(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    rows = conn.execute(
        "SELECT * FROM translation_chunks WHERE paper_id=? ORDER BY chunk_index",
        (source_id,),
    ).fetchall()
    for source in rows:
        target = conn.execute(
            "SELECT * FROM translation_chunks WHERE paper_id=? AND chunk_index=?",
            (target_id, source["chunk_index"]),
        ).fetchone()
        if not target:
            conn.execute(
                "UPDATE translation_chunks SET paper_id=? WHERE paper_id=? AND chunk_index=?",
                (target_id, source_id, source["chunk_index"]),
            )
            continue

        source_done = source["status"] in {"done", "completed", "success"} and not is_blank(source["translation"])
        target_done = target["status"] in {"done", "completed", "success"} and not is_blank(target["translation"])
        if source_done and not target_done:
            conn.execute(
                """
                UPDATE translation_chunks
                SET chunk_total=?, source_hash=?, source_label=?, source_text=?,
                    translation=?, model=?, token_count=?, status=?, error=?, updated_at=?
                WHERE paper_id=? AND chunk_index=?
                """,
                (
                    source["chunk_total"],
                    source["source_hash"],
                    source["source_label"],
                    source["source_text"],
                    source["translation"],
                    source["model"],
                    source["token_count"],
                    source["status"],
                    source["error"],
                    source["updated_at"],
                    target_id,
                    source["chunk_index"],
                ),
            )
        conn.execute(
            "DELETE FROM translation_chunks WHERE paper_id=? AND chunk_index=?",
            (source_id, source["chunk_index"]),
        )


def merge_figure_refs(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    rows = conn.execute("SELECT * FROM figure_refs WHERE paper_id=?", (source_id,)).fetchall()
    for source in rows:
        target = conn.execute(
            "SELECT * FROM figure_refs WHERE paper_id=? AND figure_no=?",
            (target_id, source["figure_no"]),
        ).fetchone()
        if not target:
            conn.execute(
                "UPDATE figure_refs SET paper_id=? WHERE paper_id=? AND figure_no=?",
                (target_id, source_id, source["figure_no"]),
            )
            continue

        if float(source["confidence"] or 0) > float(target["confidence"] or 0):
            conn.execute(
                """
                UPDATE figure_refs
                SET url=?, caption=?, page=?, first_ref_offset=?, first_ref_text=?,
                    confidence=?, updated_at=?
                WHERE paper_id=? AND figure_no=?
                """,
                (
                    source["url"],
                    source["caption"],
                    source["page"],
                    source["first_ref_offset"],
                    source["first_ref_text"],
                    source["confidence"],
                    source["updated_at"],
                    target_id,
                    source["figure_no"],
                ),
            )
        conn.execute(
            "DELETE FROM figure_refs WHERE paper_id=? AND figure_no=?",
            (source_id, source["figure_no"]),
        )


def merge_llm_jobs(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    rows = conn.execute("SELECT * FROM llm_jobs WHERE paper_id=?", (source_id,)).fetchall()
    priority = {
        "running": 5,
        "processing": 5,
        "queued": 4,
        "pending": 4,
        "completed": 3,
        "done": 3,
        "success": 3,
        "failed": 1,
        "error": 1,
    }
    for source in rows:
        target = conn.execute(
            "SELECT * FROM llm_jobs WHERE paper_id=? AND job_type=?",
            (target_id, source["job_type"]),
        ).fetchone()
        if not target:
            conn.execute(
                "UPDATE llm_jobs SET paper_id=? WHERE paper_id=? AND job_type=?",
                (target_id, source_id, source["job_type"]),
            )
            continue

        if priority.get(source["status"], 0) > priority.get(target["status"], 0):
            conn.execute(
                """
                UPDATE llm_jobs
                SET status=?, force=?, error=?, created_at=?, updated_at=?,
                    started_at=?, finished_at=?
                WHERE paper_id=? AND job_type=?
                """,
                (
                    source["status"],
                    source["force"],
                    source["error"],
                    source["created_at"],
                    source["updated_at"],
                    source["started_at"],
                    source["finished_at"],
                    target_id,
                    source["job_type"],
                ),
            )
        conn.execute(
            "DELETE FROM llm_jobs WHERE paper_id=? AND job_type=?",
            (source_id, source["job_type"]),
        )


def merge_duplicate_rows(conn: sqlite3.Connection, canonical: sqlite3.Row, duplicate: sqlite3.Row) -> None:
    target_id = canonical["id"]
    source_id = duplicate["id"]
    merge_paper_metadata(conn, canonical, duplicate)
    merge_analyses(conn, source_id, target_id)
    merge_reading_status(conn, source_id, target_id)
    conn.execute("UPDATE notes SET paper_id=? WHERE paper_id=?", (target_id, source_id))
    merge_translation_chunks(conn, source_id, target_id)
    merge_figure_refs(conn, source_id, target_id)
    merge_llm_jobs(conn, source_id, target_id)
    conn.execute("DELETE FROM papers WHERE id=?", (source_id,))


def merge_duplicate_papers(reader_db: Path, dry_run: bool = False, backup: bool = True) -> dict:
    conn = sqlite3.connect(str(reader_db))
    conn.row_factory = sqlite3.Row
    backup_path = None
    try:
        rows = conn.execute("SELECT * FROM papers").fetchall()
        groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in rows:
            groups.setdefault((row["conference"], title_key(row["title"])), []).append(row)
        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        duplicate_rows = sum(len(group) - 1 for group in duplicate_groups)
        if dry_run:
            return {
                "reader_db": str(reader_db),
                "duplicate_groups": len(duplicate_groups),
                "duplicate_rows": duplicate_rows,
                "merged_rows": 0,
                "backup": None,
                "total": len(rows),
            }

        if duplicate_rows and backup:
            backup_path = backup_database(reader_db)

        merged_rows = 0
        conn.execute("BEGIN")
        for group in duplicate_groups:
            counts = state_counts(conn, [row["id"] for row in group])
            canonical = choose_canonical(group, counts)
            for duplicate in group:
                if duplicate["id"] == canonical["id"]:
                    continue
                merge_duplicate_rows(conn, canonical, duplicate)
                merged_rows += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return {
            "reader_db": str(reader_db),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_rows": duplicate_rows,
            "merged_rows": merged_rows,
            "backup": str(backup_path) if backup_path else None,
            "total": total,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--reader-db", type=Path, default=DEFAULT_READER_DB)
    parser.add_argument("--dedupe", action="store_true", help="Merge duplicate rows by conference/title")
    parser.add_argument("--dry-run", action="store_true", help="Only report duplicate counts")
    parser.add_argument("--no-backup", action="store_true", help="Do not copy the SQLite DB before dedupe")
    args = parser.parse_args()
    result = {} if args.dry_run else import_papers(args.metadata, args.reader_db)
    if args.dedupe or args.dry_run:
        result["dedupe"] = merge_duplicate_papers(
            args.reader_db,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
