"""Bounded, atomic medical-corpus acquisition using the existing sanitized API client."""
import gzip
import hashlib
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
DATA = ROOT / "data/adoption_2026"
OUT = HERE / "results"
FILTER = "primary_topic.id:T10052|T12422,from_publication_date:2010-01-01,to_publication_date:2026-10-14,has_abstract:true,language:en"


def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pages = DATA / "pages"
    pages.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("_adoption_openalex_client", HERE.parent / "campaign/fetch_corpus.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.FILTER = FILTER
    client.SELECT += ",locations"
    cursor, page, expected, cost, network = "*", 0, None, 0., 0
    partial = DATA / ".medical.partial.jsonl.gz"
    target = DATA / "medical.jsonl.gz"
    seen = set()
    with gzip.open(partial, "wt") as stream:
        while cursor:
            cache = pages / f"{page:04d}.json.gz"
            if cache.exists():
                with gzip.open(cache, "rt") as f:
                    payload = json.load(f)
            else:
                payload = client.get_page(cursor)
                network += 1
                cost += payload.get("meta", {}).get("cost_usd", 0)
                if cost > .15:
                    raise RuntimeError("Acquisition cost cap reached; accepted corpus preserved")
                with gzip.open(cache, "wt") as f:
                    json.dump(payload, f)
                time.sleep(.15)
            count = payload["meta"]["count"]
            expected = count if expected is None else expected
            if expected > 120000 or count != expected:
                raise RuntimeError("Count cap or snapshot-count change; accepted corpus preserved")
            for work in payload["results"]:
                if work["id"] in seen:
                    continue
                seen.add(work["id"])
                compact = client.compact_work(work)
                urls = [str(location.get(field) or "") for location in work.get("locations", []) for field in ["landing_page_url", "pdf_url"]]
                compact["arxiv_ids"] = sorted({match for url in urls for match in re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z-]+/\d{7})", url, re.I)})
                compact["institutions"] = sorted({i["id"] for a in work.get("authorships", []) for i in a.get("institutions", []) if i.get("id")})
                stream.write(json.dumps(compact)+"\n")
            cursor = payload["meta"].get("next_cursor")
            page += 1
            if not payload["results"]:
                break
            if page % 25 == 0:
                print(f"Medical pages {page}; records {len(seen)}/{expected}", flush=True)
    if len(seen) != expected:
        raise RuntimeError("Incomplete acquisition; accepted corpus preserved")
    partial.replace(target)
    manifest = dict(source="OpenAlex Works API", filter=FILTER, select=client.SELECT,
        assembled_at=datetime.now(timezone.utc).isoformat(), records=len(seen), pages=page,
        network_pages=network, reported_network_cost_usd=cost,
        path=str(target.relative_to(ROOT)), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        historical_status="current metadata reconstruction; first-version audit separate")
    (OUT / "medical_acquisition.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
