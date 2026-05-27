#!/usr/bin/env python3
"""
Collect AAAI proceedings metadata from the official AAAI OJS archive.

The script discovers issue pages from the archive pagination, extracts article
metadata and PDF galley URLs, and writes ResearchKB-compatible paper records.
It does not download PDFs by default.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://ojs.aaai.org/index.php/AAAI"
ARCHIVE_URL = f"{BASE_URL}/issue/archive"
YEAR_LABEL = {2026: "AAAI-26", 2025: "AAAI-25", 2024: "AAAI-24"}


def make_session(proxy: str | None = None, no_proxy: bool = False) -> requests.Session:
    session = requests.Session()
    session.trust_env = not no_proxy
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    elif no_proxy:
        session.proxies = {"http": None, "https": None}
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/124.0 Safari/537.36 ResearchKB/AAAI"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


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


def normalize_space(value: str) -> str:
    return " ".join(html.unescape(value or "").split())


def request_soup(session: requests.Session, url: str, timeout: int, retries: int = 4) -> BeautifulSoup:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            last_error = exc
            wait = min(8 * attempt, 45)
            print(f"request failed ({attempt}/{retries}) {url}: {exc}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise last_error


def discover_issues(
    session: requests.Session,
    years: set[int],
    include_nontechnical: bool,
    max_archive_pages: int,
    timeout: int,
    delay: float,
) -> list[dict[str, str | int]]:
    wanted_labels = {year: YEAR_LABEL[year] for year in years}
    issues_by_url: dict[str, dict[str, str | int]] = {}

    for page in range(1, max_archive_pages + 1):
        archive_url = ARCHIVE_URL if page == 1 else f"{ARCHIVE_URL}/{page}"
        soup = request_soup(session, archive_url, timeout=timeout)
        found_on_page = 0

        for anchor in soup.select('a[href*="/issue/view/"]'):
            title = normalize_space(anchor.get_text(" ", strip=True))
            href = anchor.get("href")
            if not title or not href:
                continue

            for year, label in wanted_labels.items():
                if label not in title:
                    continue
                if not include_nontechnical and "Technical Tracks" not in title:
                    continue
                url = urljoin(BASE_URL, href)
                issues_by_url[url] = {"year": year, "title": title, "url": url}
                found_on_page += 1

        if found_on_page == 0 and page > 4:
            break
        time.sleep(delay)

    issues = sorted(
        issues_by_url.values(),
        key=lambda item: (int(item["year"]), str(item["title"])),
        reverse=True,
    )
    return issues


def manual_issues(issue_ids: list[int], year: int) -> list[dict[str, str | int]]:
    return [
        {
            "year": year,
            "title": f"AAAI-{str(year)[-2:]} Manual Issue {issue_id}",
            "url": f"{BASE_URL}/issue/view/{issue_id}",
        }
        for issue_id in issue_ids
    ]


def parse_issue_articles(
    session: requests.Session,
    issue: dict[str, str | int],
    timeout: int,
) -> list[dict[str, str | int | dict | list]]:
    soup = request_soup(session, str(issue["url"]), timeout=timeout)
    issue_title = str(issue["title"])
    year = int(issue["year"])
    venue = f"AAAI {year}"
    papers = []

    for article in soup.select(".obj_article_summary"):
        title_anchor = article.select_one("h3.title a")
        if not title_anchor:
            continue

        title = normalize_space(title_anchor.get_text(" ", strip=True))
        article_url = urljoin(BASE_URL, title_anchor.get("href", ""))
        article_id_match = re.search(r"/article/view/(\d+)", article_url)
        article_id = article_id_match.group(1) if article_id_match else hashlib.md5(article_url.encode()).hexdigest()[:12]
        authors_text = normalize_space(
            article.select_one(".authors").get_text(" ", strip=True)
            if article.select_one(".authors")
            else ""
        )
        authors = [normalize_space(author) for author in authors_text.split(",") if normalize_space(author)]
        pages = normalize_space(
            article.select_one(".pages").get_text(" ", strip=True)
            if article.select_one(".pages")
            else ""
        )
        pdf_url = ""
        for galley in article.select("a.obj_galley_link"):
            if "pdf" in " ".join(galley.get("class", [])).lower() or "PDF" in galley.get_text(" ", strip=True):
                pdf_url = urljoin(BASE_URL, galley.get("href", ""))
                break

        paper_id = f"aaai_{year}_{article_id}"
        papers.append(
            {
                "paperId": paper_id,
                "title": title,
                "abstract": "",
                "year": str(year),
                "venue": venue,
                "citationCount": 0,
                "doi": "",
                "dblp_key": "",
                "openAccessPdf": {"url": pdf_url} if pdf_url else None,
                "externalIds": {"AAAI": article_id},
                "authors": authors,
                "pages": pages,
                "_source_conference": venue,
                "_source_issue": issue_title,
                "_source_url": article_url,
                "_search_keyword": "AAAI OJS",
            }
        )
    return papers


def merge_papers(existing: list[dict], new_papers: list[dict]) -> tuple[list[dict], int, int]:
    by_id = {paper.get("paperId"): paper for paper in existing if isinstance(paper, dict) and paper.get("paperId")}
    added = 0
    updated = 0
    for paper in new_papers:
        paper_id = paper["paperId"]
        if paper_id in by_id:
            by_id[paper_id].update({k: v for k, v in paper.items() if v not in ("", None, [], {})})
            updated += 1
        else:
            by_id[paper_id] = paper
            added += 1
    return list(by_id.values()), added, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AAAI OJS metadata")
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    parser.add_argument("--years", nargs="+", type=int, default=[2026, 2025, 2024])
    parser.add_argument("--include-nontechnical", action="store_true")
    parser.add_argument("--issue-ids", nargs="+", type=int, default=None)
    parser.add_argument("--issue-year", type=int, default=None)
    parser.add_argument("--max-archive-pages", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    unknown = sorted(set(args.years) - set(YEAR_LABEL))
    if unknown:
        raise SystemExit(f"unsupported AAAI years: {unknown}")

    session = make_session(proxy=args.proxy, no_proxy=args.no_proxy)
    if args.proxy:
        print(f"proxy={args.proxy}", flush=True)
    elif args.no_proxy:
        print("proxy=disabled", flush=True)
    else:
        print("proxy=env", flush=True)
    if args.issue_ids:
        issue_year = args.issue_year or args.years[0]
        issues = manual_issues(args.issue_ids, issue_year)
    else:
        issues = discover_issues(
            session=session,
            years=set(args.years),
            include_nontechnical=args.include_nontechnical,
            max_archive_pages=args.max_archive_pages,
            timeout=args.timeout,
            delay=args.delay,
        )
    print(f"issues={len(issues)}")
    for issue in issues[:10]:
        print(f"  {issue['year']} {issue['title']} {issue['url']}", flush=True)
    if len(issues) > 10:
        print(f"  ... {len(issues) - 10} more", flush=True)

    all_new = []
    for issue in issues:
        papers = parse_issue_articles(session, issue, timeout=args.timeout)
        print(f"{issue['title']}: {len(papers)} papers", flush=True)
        all_new.extend(papers)
        time.sleep(args.delay)

    print(f"collected={len(all_new)}", flush=True)
    if args.output:
        save_json(args.output, all_new)
        print(f"output={args.output}", flush=True)
        return

    if args.dry_run:
        sample_path = args.kb_root / "metadata" / "aaai_ojs_sample.json"
        save_json(sample_path, all_new[:50])
        print(f"dry_run_sample={sample_path}", flush=True)
        return

    metadata_path = args.kb_root / "metadata" / "all_papers.json"
    existing = load_json(metadata_path, [])
    merged, added, updated = merge_papers(existing, all_new)
    save_json(metadata_path, merged)
    print(f"metadata={metadata_path}")
    print(f"added={added} updated={updated} total={len(merged)}")


if __name__ == "__main__":
    main()
