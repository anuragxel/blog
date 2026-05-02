#!/usr/bin/env python3
"""
Fill in `url={...}` for entries in _bibliography/references.bib that
don't have one.

Resolution order per entry:
  1. arXiv ID embedded in journal / note / booktitle / eprint fields
  2. Semantic Scholar API by title (uses ArXiv id if found, else paper URL)

Existing `url=` fields are never touched — paste a project page URL into
the bib by hand and it overrides everything below it.

Usage:  make resolve-cites
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import bibtexparser
from bibtexparser.bwriter import BibTexWriter

BIB_PATH = Path("_bibliography/references.bib")
SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
RATE_LIMIT_S = 3.5     # Semantic Scholar unauthed limit is ~1 rps but bursty.
TIMEOUT_S = 15
MAX_RETRIES = 3

ARXIV_LABEL_RE = re.compile(r"arXiv[:\s]*?(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def find_arxiv_id(*texts: str | None) -> str | None:
    for t in texts:
        if not t:
            continue
        m = ARXIV_LABEL_RE.search(t) or ARXIV_BARE_RE.search(t)
        if m:
            return m.group(1)
    return None


def query_semantic_scholar(title: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "query": title,
            "limit": 1,
            "fields": "url,externalIds,openAccessPdf,title",
        }
    )
    req = urllib.request.Request(
        f"{SS_API}?{params}",
        headers={"Accept": "application/json", "User-Agent": "blog-bib-resolver/1.0"},
    )
    data = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read())
                break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                backoff = 5 * (attempt + 1)
                print(f"    [rate-limited] backing off {backoff}s...", file=sys.stderr)
                time.sleep(backoff)
                continue
            print(f"    [warn] semantic scholar: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"    [warn] semantic scholar: {e}", file=sys.stderr)
            return None
    if data is None:
        return None

    items = data.get("data") or []
    if not items:
        return None
    p = items[0]

    arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    pdf = (p.get("openAccessPdf") or {}).get("url")
    if pdf:
        return pdf

    return p.get("url")


def main() -> int:
    if not BIB_PATH.exists():
        print(f"error: {BIB_PATH} not found (run from repo root)", file=sys.stderr)
        return 1

    with BIB_PATH.open() as f:
        db = bibtexparser.load(f)

    changed = 0
    for entry in db.entries:
        key = entry.get("ID", "?")

        if entry.get("url"):
            print(f"  = {key}: already has url")
            continue

        arxiv_id = find_arxiv_id(
            entry.get("journal"),
            entry.get("note"),
            entry.get("booktitle"),
            entry.get("eprint"),
        )
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
            entry["url"] = url
            print(f"  + {key}: {url} (arxiv id)")
            changed += 1
            continue

        title = (entry.get("title") or "").strip("{}").strip()
        if not title:
            print(f"  ? {key}: no title to look up")
            continue

        url = query_semantic_scholar(title)
        if url:
            entry["url"] = url
            print(f"  + {key}: {url}")
            changed += 1
        else:
            print(f"  ? {key}: no url found (add manually)")
        time.sleep(RATE_LIMIT_S)

    if changed:
        writer = BibTexWriter()
        writer.indent = "  "
        writer.add_trailing_comma = False
        writer.order_entries_by = None        # preserve original entry order
        writer.display_order = ("title", "author", "booktitle", "journal", "year", "url")
        with BIB_PATH.open("w") as f:
            f.write(writer.write(db))
        print(f"\nupdated {changed} entr{'y' if changed == 1 else 'ies'} in {BIB_PATH}")
    else:
        print("\nno changes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
