#!/usr/bin/env python3
"""Download OJS PDF galleys for metadata records."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]


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
                "Chrome/124.0 Safari/537.36 ResearchKB/OJS-PDF"
            ),
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://ojs.aaai.org/index.php/AAAI/",
        }
    )
    return session


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def download_pdf(session: requests.Session, url: str, timeout: int, retries: int) -> bytes | None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            if response.status_code == 200 and response.content[:4] == b"%PDF":
                return response.content
            raise requests.HTTPError(
                f"not pdf status={response.status_code} content_type={response.headers.get('content-type')}"
            )
        except Exception as exc:
            last_error = exc
            wait = min(5 * attempt, 30)
            print(f"download failed ({attempt}/{retries}) {url}: {exc}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AAAI/OJS PDFs")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--download-map", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()

    papers = load_json(args.metadata, [])
    download_map = load_json(args.download_map, {})
    session = make_session(args.proxy, args.no_proxy)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    targets = [p for p in papers if str(p.get("paperId", "")).startswith("aaai_")]
    if args.limit:
        targets = targets[: args.limit]

    ok = skip = fail = 0
    failures = []
    total = len(targets)
    for idx, paper in enumerate(targets, start=1):
        paper_id = paper["paperId"]
        if paper_id in download_map and Path(download_map[paper_id]).exists():
            skip += 1
            continue
        url = (paper.get("openAccessPdf") or {}).get("url")
        if not url:
            fail += 1
            failures.append({"paperId": paper_id, "reason": "missing url"})
            continue
        path = args.pdf_dir / f"{paper_id}.pdf"
        if path.exists() and path.stat().st_size > 1000:
            download_map[paper_id] = str(path)
            skip += 1
            continue
        content = download_pdf(session, url, args.timeout, args.retries)
        if content:
            path.write_bytes(content)
            download_map[paper_id] = str(path)
            ok += 1
        else:
            fail += 1
            failures.append({"paperId": paper_id, "url": url})
        if idx % 20 == 0 or idx == total:
            save_json(args.download_map, download_map)
            print(f"pdf {idx}/{total} ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(args.delay)

    save_json(args.download_map, download_map)
    if failures:
        save_json(args.download_map.parent / "ojs_pdf_failures.json", failures)
    print(f"done ok={ok} skip={skip} fail={fail} map={len(download_map)}", flush=True)


if __name__ == "__main__":
    main()
