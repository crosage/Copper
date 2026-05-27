#!/usr/bin/env python3
"""Prepare merged metadata for ResearchKB v2 without modifying the legacy files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ResearchKB v2 merged metadata")
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_KB_ROOT / "v2_source")
    parser.add_argument("--extra-metadata", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    base_papers = load_json(args.kb_root / "metadata" / "all_papers.json", [])
    base_download_map = load_json(args.kb_root / "metadata" / "download_map.json", {})
    by_id = {paper.get("paperId"): paper for paper in base_papers if isinstance(paper, dict) and paper.get("paperId")}

    extra_count = 0
    for path in args.extra_metadata:
        for paper in load_json(path, []):
            if not isinstance(paper, dict) or not paper.get("paperId"):
                continue
            by_id[paper["paperId"]] = paper
            extra_count += 1

    out_root = args.out_root.resolve()
    save_json(out_root / "metadata" / "all_papers.json", list(by_id.values()))
    save_json(out_root / "metadata" / "download_map.json", base_download_map)
    print(f"base={len(base_papers)} extra={extra_count} merged={len(by_id)}")
    print(f"metadata={out_root / 'metadata' / 'all_papers.json'}")
    print(f"download_map={out_root / 'metadata' / 'download_map.json'}")


if __name__ == "__main__":
    main()
