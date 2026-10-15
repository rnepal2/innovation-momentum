"""Acquire a complete bounded OpenAlex corpus with resumable cursor pages."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "data" / "research_2026"
FILTER = (
    "primary_topic.id:T10462|T10653,from_publication_date:2010-01-01,"
    "to_publication_date:2025-12-31,has_abstract:true,language:en"
)
SELECT = "id,doi,title,publication_date,primary_topic,authorships,abstract_inverted_index,referenced_works"


def get_page(cursor: str) -> dict:
    params = {"filter": FILTER, "select": SELECT, "per-page": 200, "cursor": cursor}
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "innovation-momentum-research/0.2"}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise RuntimeError(f"OpenAlex request failed with HTTP {error.code}") from None
            print(f"Retry {attempt + 1}: HTTP {error.code}", flush=True)
            time.sleep(min(5 * 2**attempt, 30))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise RuntimeError("OpenAlex request failed after connection retries") from None
            print(f"Retry {attempt + 1}: connection failure", flush=True)
            time.sleep(min(5 * 2**attempt, 30))
    raise RuntimeError("Unreachable")


def reconstruct_abstract(index: dict | None) -> str:
    if not index:
        return ""
    positions = [
        (position, word) for word, occurrences in index.items() for position in occurrences
    ]
    return " ".join(word for _, word in sorted(positions))


def compact_work(work: dict) -> dict:
    return {
        "id": work["id"].rsplit("/", 1)[-1],
        "doi": work.get("doi"),
        "title": work.get("title") or "",
        "date": work["publication_date"],
        "topic": (work.get("primary_topic") or {}).get("id", "").rsplit("/", 1)[-1],
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "authors": [
            a["author"]["id"].rsplit("/", 1)[-1]
            for a in work.get("authorships", [])
            if a.get("author", {}).get("id")
        ],
        "references": [
            reference.rsplit("/", 1)[-1] for reference in work.get("referenced_works", [])
        ],
    }


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    pages = DEST / "pages"
    pages.mkdir(exist_ok=True)
    cursor = "*"
    page_number = 0
    total = 0
    seen = set()
    output = DEST / "robotics_works.jsonl.gz"
    partial = DEST / ".robotics_works.partial.jsonl.gz"
    expected = None
    network_pages = 0
    with gzip.open(partial, "wt", encoding="utf-8") as stream:
        while cursor:
            path = pages / f"{page_number:04d}.json.gz"
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as cached:
                    payload = json.load(cached)
            else:
                payload = get_page(cursor)
                network_pages += 1
                with gzip.open(path, "wt", encoding="utf-8") as cached:
                    json.dump(payload, cached)
                time.sleep(0.15)
            count = payload["meta"]["count"]
            if expected is None:
                expected = count
            elif count != expected:
                raise RuntimeError(
                    "Corpus count changed during pagination; preserve the accepted corpus and inspect cached pages"
                )
            for work in payload.get("results", []):
                if work["id"] in seen:
                    continue
                seen.add(work["id"])
                stream.write(json.dumps(compact_work(work), ensure_ascii=True) + "\n")
                total += 1
            cursor = payload.get("meta", {}).get("next_cursor")
            if not payload.get("results"):
                break
            page_number += 1
            if page_number % 10 == 0:
                print(
                    f"Pages {page_number}; unique works {total}; expected {payload['meta']['count']}",
                    flush=True,
                )
    if total != expected:
        raise RuntimeError(
            f"Incomplete corpus: retained {total} unique records; expected {expected}"
        )
    partial.replace(output)
    manifest = {
        "source": "OpenAlex Works API",
        "filter": FILTER,
        "select": SELECT,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "network_pages_this_run": network_pages,
        "cache_last_modified_at": datetime.fromtimestamp(
            max(p.stat().st_mtime for p in pages.glob("*.gz")), timezone.utc
        ).isoformat(),
        "works": total,
        "pages": page_number,
        "path": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "historical_status": "retrospective current-metadata reconstruction",
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
