"""Harvest arXiv robotics metadata using the documented, rate-limited OAI interface."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "data" / "research_2026"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
ENDPOINT = "https://oaipmh.arxiv.org/oai"


def author_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", normalized.lower())


def first_month(identifier: str) -> str:
    digits = identifier.split("/")[-1][:4]
    year = int(digits[:2])
    return f"{1900 + year if year >= 91 else 2000 + year:04d}-{digits[2:4]}-01"


def parse_records(root: ET.Element) -> list[dict]:
    records = []
    for record in root.findall(".//o:record", NS):
        metadata = record.find("o:metadata/a:arXiv", NS)
        if metadata is None:
            continue

        def value(key):
            return metadata.findtext(f"a:{key}", "", NS).strip()

        identifier = value("id")
        month = first_month(identifier)
        if not "2010" <= month[:4] <= "2025":
            continue
        authors = []
        for author in metadata.findall("a:authors/a:author", NS):
            given = author.findtext("a:forenames", "", NS)
            surname = author.findtext("a:keyname", "", NS)
            key = author_key(f"{given} {surname}")
            if key:
                authors.append(key)
        records.append(
            {
                "id": identifier,
                "doi": value("doi"),
                "title": " ".join(value("title").split()),
                "date": month,
                "metadata_created": value("created"),
                "metadata_updated": value("updated"),
                "topic": "cs.RO",
                "categories": value("categories").split(),
                "abstract": " ".join(value("abstract").split()),
                "authors": authors,
                "references": [],
            }
        )
    return records


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    pages = DEST / "arxiv_pages"
    pages.mkdir(exist_ok=True)
    token = None
    page = 0
    total = 0
    harvested = 0
    last_request = 0.0
    output = DEST / "arxiv_robotics.jsonl.gz"
    partial = DEST / ".arxiv_robotics.partial.jsonl.gz"
    network_pages = 0
    seen = set()
    with gzip.open(partial, "wt", encoding="utf-8") as stream:
        while True:
            path = pages / f"{page:04d}.xml.gz"
            if path.exists():
                with gzip.open(path, "rb") as cached:
                    payload = cached.read()
            else:
                params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs:cs:RO"}
                if token:
                    params = {"verb": "ListRecords", "resumptionToken": token}
                url = ENDPOINT + "?" + urllib.parse.urlencode(params)
                for attempt in range(4):
                    time.sleep(max(0.0, 3.1 - (time.monotonic() - last_request)))
                    last_request = time.monotonic()
                    try:
                        with urllib.request.urlopen(url, timeout=90) as response:
                            payload = response.read()
                        network_pages += 1
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if attempt == 3:
                            raise
                        time.sleep(10 * (attempt + 1))
                with gzip.open(path, "wb") as cached:
                    cached.write(payload)
            root = ET.fromstring(payload)
            error = root.find("o:error", NS)
            if error is not None:
                raise RuntimeError(f"OAI error: {error.attrib}: {error.text}")
            harvested += len(root.findall(".//o:record", NS))
            for record in parse_records(root):
                if record["id"] not in seen:
                    seen.add(record["id"])
                    stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                    total += 1
            page += 1
            print(
                f"OAI page {page}; harvested {harvested}; eligible unique papers {total}",
                flush=True,
            )
            element = root.find(".//o:resumptionToken", NS)
            token = element.text if element is not None else None
            if not token:
                break
    partial.replace(output)
    manifest = {
        "source": "arXiv OAI-PMH",
        "endpoint": ENDPOINT,
        "set": "cs:cs:RO",
        "metadata_prefix": "arXiv",
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "network_pages_this_run": network_pages,
        "cache_last_modified_at": datetime.fromtimestamp(
            max(p.stat().st_mtime for p in pages.glob("*.gz")), timezone.utc
        ).isoformat(),
        "filter": "arXiv identifier first-submission month within 2010-2025, nondeleted metadata",
        "papers": total,
        "harvested_records": harvested,
        "pages": page,
        "path": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "author_identity": "normalized full name; unresolved homonyms and name variants",
        "text_status": "latest abstract metadata; retrospective text reconstruction",
        "references_available": False,
    }
    (DEST / "arxiv_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
